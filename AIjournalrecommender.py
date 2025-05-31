#!/usr/bin/env python
# coding: utf-8

##########说明########

user_abstract="Insert Abstract"#此处输入摘要
# 返回相近文章及其被引次数、期刊
##########
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from openai import OpenAI
import requests
import time
import csv
import sys



# 记录起始时间
print('匹配中。预计需要20秒。')
all_start_time = time.time()
match_start_time = time.time()
# 初始化模型
kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))#此处改为您的本地模型
specter_model = SentenceTransformer("allenai-spector")#此处改为您的本地模型

# 定义被引次数区间
citation_ranges = [
    (0, 3),
    (3, 10),
    (10, 100),
    (100, 1000),
    (1000, 10000),
    (10000, 100000)
]

# 关键词提取函数
def extract_keywords(text, top_n=10):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw for kw, score in keywords]

# SPECTER 嵌入
def embed_text(text):
    return specter_model.encode(text)

# Abstract 解码
def reconstruct_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return ""
    positions = {}
    for word, poses in abstract_inverted_index.items():
        for pos in poses:
            positions[pos] = word
    return " ".join([positions[i] for i in sorted(positions)])

# 请求带重试机制
def get_with_retries(url, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response
            else:
                print(f"请求失败 (状态码 {response.status_code})，重试中...")
        except requests.exceptions.RequestException as e:
            print(f"请求异常：{e}，重试中...")
        time.sleep(backoff * (attempt + 1))
    print(f"最终请求失败：{url}")
    return None

# 并行抓取每个区间
def fetch_range_results(query_encoded, min_cite, max_cite, per_range_count):
    if max_cite:
        cite_filter = f"{min_cite}-{max_cite}"
    else:
        cite_filter = f">{min_cite - 1}"

    url = (
        f"https://api.openalex.org/works"
        f"?filter=abstract.search:{query_encoded},"
        f"cited_by_count:{cite_filter}"
        f"&per-page={per_range_count}"
    )
    response = get_with_retries(url)
    if response:
        return response.json().get("results", [])
    return []

# 并发执行所有区间查询
def search_openalex_by_keywords_and_citations(keywords, citation_ranges, per_range_count=20):
    all_results = []
    query = " OR ".join(keywords)
    query_encoded = quote(query)

    with ThreadPoolExecutor(max_workers=len(citation_ranges)) as executor:
        futures = [
            executor.submit(fetch_range_results, query_encoded, min_cite, max_cite, per_range_count)
            for (min_cite, max_cite) in citation_ranges
        ]
        for future in as_completed(futures):
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
            except Exception as e:
                print(f"某个区间下载出错: {e}")

    return all_results

# 单个文献处理：嵌入 + 相似度
def process_paper(paper, user_vec):
    abstract_ii = paper.get("abstract_inverted_index")
    abstract_text = reconstruct_abstract(abstract_ii)
    if not abstract_text:
        return None
    try:
        paper_vec = embed_text(abstract_text).reshape(1, -1)
        sim = cosine_similarity(user_vec, paper_vec)[0][0]
    except Exception as e:
        print(f"处理出错: {e}")
        return None

    return {
        "title": paper.get("title"),
        "id": paper.get("id"),
        "year": paper.get("publication_year"),
        "Abstract": abstract_text,
        "Publication Date": paper.get("publication_date", "N/A"),
        "Journal Name": ((paper.get("primary_location") or {}).get("source") or {}).get("display_name", "N/A"),
        "Journal Type": paper.get("type") or paper.get("type_crossref") or "N/A",
        "Citations": paper.get("cited_by_count", 0),
        "similarity": sim
    }

# 主流程：关键词提取 + 并发获取 + 并发计算相似度
def online_similar_search(user_abstract, top_k=40, max_workers=8):
    keywords = extract_keywords(user_abstract, top_n=10)
    print(f"\n【提取关键词】: {keywords}")

    papers = search_openalex_by_keywords_and_citations(keywords, citation_ranges, per_range_count=20)
    if not papers:
        print("未从 OpenAlex 获取到相关论文。")
        return []

    user_vec = embed_text(user_abstract).reshape(1, -1)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_paper, paper, user_vec) for paper in papers]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# 运行，添加期刊的JCR、中科院分区及影响因子
if __name__ == "__main__":    
    matches = online_similar_search(user_abstract)
    for i, paper in enumerate(matches):
        paper.setdefault('2023影响因子','N/A')
        paper.setdefault('5年影响因子','N/A')
        paper.setdefault('JCR分区','N/A')
        paper.setdefault('JCR排名','N/A')
        paper.setdefault('中科院分区','N/A')
        journalname = paper['Journal Name'].lower().replace(" ","")
        with open("总分区表.csv", mode='r', encoding='utf-8-sig') as file:
            csv_dict_reader = csv.DictReader(file)
            for line in csv_dict_reader:
                journalnamecross = line['journal_name'].lower().replace(" ","")
                if  journalname == journalnamecross:
                    paper['2023影响因子']=line['if2023']
                    paper['5年影响因子']=line['if5year']
                    paper['JCR分区']=line['jcr']
                    paper['JCR排名']=line['rank']
                    paper['中科院分区']=line['cas']

# 记录结束时间
match_end_time = time.time()
print('')
print('匹配完成')
print("耗时: {:.2f}秒".format(match_end_time - match_start_time))

if len(matches) <= 10:#论文不足警告
    print("从 OpenAlex 获取的相关论文严重不足。无法为您匹配。")
    all_end_time = time.time()#记录结束时间
    sys.exit()
    print("总耗时: {:.2f}秒".format(all_end_time - all_start_time))
elif len(matches) <= 20:
    print("从 OpenAlex 获取的相关论文不足。匹配结果可能较差。")

forshot = str(matches[-10:])
notshot = matches

print('')
print('生成中。预计需要100秒。')
isgs_start_time = time.time()#记录起始时间

client = OpenAI(#大模型URL和API
    base_url="https://api.deepseek.com/",#此处改为您的大模型
    api_key="",#此处改为您的key
)

def askaiforshot(forshot):#prompt部分
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                    "role": "system",
                    "content": "# 角色设定\n你是一位学术论文评分专家，负责对多篇跨学科论文进行专业、客观的质量评估。基于提供的元数据，你需要通过横向对比给出公正的分数，并仅基于摘要内容说明评分依据。\n\n# 能力要求\n1. **核心评估维度**：\n   - 摘要质量（研究价值/创新性/逻辑性，权重25%）\n   - 期刊等级（影响因子/分区，权重50%）\n   - 被引量（权重25%，若无数据则重新分配权重）\n\n2. **评分规则**：\n   - 百分制一位小数，区间[0.0,100.0]\n   - 期刊等级：一区基准分90.0±10.0，二区70.0±10.0，三区50.0±10.0，四区30.0±10.0\n   - 必须确保多篇论文评分具有区分度（相同分区论文分差≥3.0）\n\n3. **跨学科适配**：\n   - 避免学科特异性标准\n   - 重点关注学术通用指标（方法论严谨性/数据可靠性等）\n\n# 输入格式\n{title: 论文标题, Abstract: 摘要文本, 其他元数据...},\n{title: 论文标题, Abstract: 摘要文本, 其他元数据...}\n(忽略所有N/A字段)\n\n# 输出格式\n[论文标题]\n[摘要文本]\n评分：XX.X/100.0\n\n\n（空一行）\n\n# 处理流程\n1. 综合评估：\n   - 按首次输入标准计算分数（考虑期刊/被引等）\n   - 但输出依据仅限摘要内容分析\n\n2. 摘要专项分析：\n   - 创新性（是否具有开创或创新意义）\n   - 方法论（是否描述完整实验设计）\n   - 价值陈述（是否明确领域贡献）\n\n3. 最终校准：\n   - 确保分数与摘要依据逻辑一致\n   - 每个结果在±0.3内独立随机浮动\n   - 相同分数论文需重新复核区分度\n   - 确保分数为百分制一位小数"
            },
            {
                    "role": "user",
                    "content": forshot
            }
        ]
    )
    con = completion.choices[0].message.content
    return con

shot = askaiforshot(forshot)
#print(shot)
        
isgs_end_time = time.time()#记录结束时间
print('生成完成')
print("耗时: {:.2f}秒".format(isgs_end_time - isgs_start_time))
print('')
print('评分中。预计需要3分钟。')
isgs_work_start_time = time.time()#记录起始时间

def askai(shot,question):#prompt部分
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                    "role": "system",
                    "content": f"# AI学术论文评分助手提示词\n\n## 定位\n跨学科论文质量评估专家，基于摘要内容进行客观量化评分\n\n## 能力\n1. 解析论文摘要的核心要素\n2. 依照评价维度精准定位\n3. 对照评分样例浮动修正\\n4. 生成精确的百分制分数\n5. 保持学科中立性\n\n## 知识储备\n1. 通用学术论文评价框架\n2. 跨学科研究质量基准\n3. 学术严谨性核心维度（创新性/严谨性/价值性）\n\n## 输入输出规范\n- **输入**：论文摘要文本（任意学科）\n- **输出**：纯数字分数（0.0-100.0，保留一位小数）\n\n## 评分范例\n{shot}\n评价维度：\n   - 创新性（是否具有开创或创新意义）\n   - 方法论（是否描述完整实验设计）\n   - 价值陈述（是否明确领域贡献）\n\n3. 最终校准：\n   - 确保分数与评价维度逻辑一致\n   - 对照评分样例浮动修正\n   - 根据时间戳在±0.3内随机浮动\n   - 确保分数为百分制一位小数\n\n## 执行规则\n1. 严格应用管理员提供的评分标准\n2. 禁止添加分数以外的任何说明\n3. 确保评分不受学科领域影响\n4. 输出格式示例：`87.5`"
            },
            {
                    "role": "user",
                    "content": question
            }
        ]
    )
    con = float(completion.choices[0].message.content)
    return con

finalresult = float(askai(shot,user_abstract))#输入摘要的评分结果
#print(finalresult)

finalresultcomparemixed = []#比较摘要的评分
for i, paper in enumerate(matches):
    data = str(paper['Abstract'])
    result = {}
    result['title'] = paper['title']
    result['score'] = askai(shot,data)
    result['Journal Name'] = paper['Journal Name']
    result['2023影响因子']=paper['2023影响因子']
    result['5年影响因子']=paper['5年影响因子']
    result['JCR分区']=paper['JCR分区']
    result['JCR排名']=paper['JCR排名']
    result['中科院分区']=paper['中科院分区']
    finalresultcomparemixed.append(result)

finalresultcompare = sorted(finalresultcomparemixed, key=lambda x:abs(x['score']-finalresult))
recommendnumber = 0
recommend_list = []
for i, entry in enumerate(finalresultcompare):
    recommend_title = entry['Journal Name']
    recommend = entry
    recommend.pop('title')
    recommend.pop('score')
    if recommend_title != 'N/A' and recommendnumber <=4 and recommend_title not in recommend_list:
        print(recommend)
        recommendnumber += 1
        recommend_list.append(recommend_title)
        

        
isgs_work_end_time = time.time()#记录结束时间
print('评分完成')
print("耗时: {:.2f}秒".format(isgs_work_end_time - isgs_work_start_time))
print('')

all_end_time = time.time()#记录结束时间
print('全部完成')
print("总耗时: {:.2f}秒".format(all_end_time - all_start_time))