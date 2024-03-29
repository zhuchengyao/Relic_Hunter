

reader_prompt = """请你仔细阅读相关内容，结合历史资料进行回答,每一条史资料使用'出处：《书名》原文内容'的形式标注 
(如果回答请清晰无误地引用原文,先给出回答，再贴上对应的原文，使用《书名》[]对原文进行标识),如果发现资料无法得到答案,就回答不知道 \n
搜索的相关历史资料如下所示.\n
---------------------\n
{context_str}\n
---------------------\n
问题: {query_str}\n
答案: """


refiner_prompt = """你是一个历史知识回答修正机器人，你严格按以下方式工作
1.只有原答案为不知道时才进行修正,否则输出原答案的内容\n
2.修正的时候为了体现你的精准和客观，你非常喜欢使用《书名》[]将原文展示出来.\n
3.如果感到疑惑的时候，就用原答案的内容回答。\n
新的知识: {context_msg}\n
问题: {query_str}\n
原答案: {existing_answer}\n
新答案: """

