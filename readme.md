# AI期刊投递推荐助手 AI Journal Recommender 

南京大学2025年春季课程“人工智能+产品：创意、设计与开发实践”的小组课程作业。   
This is a group class project for the Spring 2025 course "AI + Products: Creativity, Design and Development Practices" at Nanjing University.    

@shenzhenyu-c、@fishoook、@ChestnutZhao（依中文姓氏拼音排序）对本人工智能工具做出了**同等贡献**。   
@shenzhenyu-c, @fishoook and @ChestnutZhao (Sorted by phonetic initials of last names in Chinese) made **equal contributions** for this AI tool.    

本工具**原样提供，不负责任**。   
This tool is presented **AS IS**, and the authors **SHALL NOT BE HELD LIABLE**.   

本人工智能工具旨在为您输入的论文摘要寻找**在主题和水平方面合适**的期刊。本人工智能工具适用于**各学科**论文。目前，本人工智能工具对**高水平**的期刊运行更好。    
This AI tool aims at matching your scientific paper abstract with **suitable journals, both subject- and level- wise**. This AI tool works for scientific papers of **all subjects**. Currently, this AI tool works the best for **high-level papers**.   
 
本工具的工作流程为：   
The working procedure of this tool is:   
①用本地部署模型，将输入的摘要选择关键词；  
①Keyword picking from the abstract input using a local model;   
②在OpenAlex上依据选择的关键词进行检索；   
②Search accoring to the picked keywords at OpenAlex;   
③将OpenAlex返回的摘要与原摘要匹配，用另一本地部署模型，按相似度排序；   
③Match the abstracts found at OpenAlex with the original abstract using another local model, and Rank according to similarity;   
④添加关于文章所在期刊影响力的元数据；   
④Add the metadata about the influence of the journals of the papers;   
⑤用Deepseek V3，将十(10)篇摘要**及其他元数据同时**进行第一次评分，**原位生成提示词**；   
⑤First Rating based on TEN(10) abstracts **and other metadata** (**simultaneous input**) using Deepseek V3, **in situ generating prompt**;   
⑥将第一次评分的结果作为系统提示词，对所有摘要**依次**进行第二次评分，**不包括元数据**；   
⑥Second Rating based on all abstracts **without metadata** (**one-by-one input**) using Deepseek V3, with the first rating results as system prompt.   
⑦返回在第二次评分中与原摘要得分最接近的五(5)篇摘要所在的期刊。   
⑦Return the FIVE(5) journals of corresponding abstracts with the closest second rating results to the original abstract.   

要使用本工具，请将fullwork.py的第6行改为您的摘要；第27、28行改为您的本地模型路径；第204、205行改为您的deepseek（或其他AI模型）链接和key，并在python控制台中运行。   
In order to use this tool, please: In the file fullwork.py, change Line 6 to your abstract, Lines 27,28 to your local models filepath, Lines 204,205 to your deepseek (or other AI models) URL link and key. Run the file in a python console.   
