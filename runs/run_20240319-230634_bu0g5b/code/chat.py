from agentscope.agents import DialogAgent, UserAgent
import agentscope
from agentscope.message import Msg
from Scope.prompt import reader_prompt, refiner_prompt
from langchain_community.vectorstores import Chroma
from Scope.utility import embedding_method, book_name


def rag(prompt_origin):
    db = Chroma(persist_directory="./Scope/db_history", embedding_function=embedding_method())
    query = prompt_origin
    res = db.similarity_search(query, k=6)
    # print(res)
    contents = [i.page_content for i in res]
    sources = [i.metadata['source'] for i in res]
    return contents, sources


def history_chatter(prompt):
    agentscope.init(model_configs="./Scope/agentscope_config.json")

    query_str = prompt
    content_str = ''
    # agent_question_name = DialogAgent(name="name_checker",
    #                              model_config_name="qwen_1.5_72B",
    #                              sys_prompt="你是我的问题提炼机器人，帮我将问题中的关键人物信息提炼出来。并直接输出,不要加任何前缀冒号:")

    agent_question_incident = DialogAgent(name="incidence_checker",
                                 model_config_name="qwen_1.5_72B",
                                 sys_prompt="提炼问题机器人，不要回答问题，而只是将问题中的最重要的历史事件直接以词的形式直接输出。"
                                            "直接回答问题，不要说任何多余的话")
    # key_element_name = agent_question_name(prompt)
    incident_prompt = "问题是：" + prompt + "你认为能代表这个问题中的事件的词是："
    key_element_incident = agent_question_incident(incident_prompt)
    # name_element = key_element_name['content']
    incident_element = key_element_incident['content']

    contents, sources = rag(incident_element)
    agent1 = DialogAgent(name="Reader",
                         model_config_name="qwen_1.5_72B",
                         sys_prompt="你是一个严谨的历史知识问答智能体，你会仔细阅读历史材料并给出准确的回答,你的回答都会非常准确，因"
                                    "为你在回答的之后，使用在《书名》[]内给出原文用来支撑你回答的证据.并且你会在开头说明原文是否有回"
                                    "答所需的知识\n")


    answers_from_agent1 = "原问题:{" + prompt + "}"
    for i, content in enumerate(contents):
        print(content)
        content_str = content
        source = book_name(sources[i])
        content_str = content_str + source
        # print(content_str)
        # print(source)
        reader_prompt = (f"请你仔细阅读相关内容，结合历史资料进行回答,每一条史资料使用'出处：《书名》原文内容'的形式标注"
                         f"(如果回答请清晰无误地引用原文,先给出回答，再贴上对应的原文，使用《书名》[]对原文进行标识),，如果发现资料"
                         f"无法得到答案,就回答不知道 \n搜索的相关历史资料如下所示.\n"
                         f"---------------------\n"
                         f"{content_str}\n"
                         f"---------------------\n"
                         f"问题: {query_str}\n"
                         f"答案: ")

        x = Msg(name="Questioner", content=reader_prompt)
        agent1_answer = agent1(x)
        answers_from_agent1=answers_from_agent1 + agent1_answer['content'] + ".\n"
        # print(agent1_answer['content'])

    agent2 = DialogAgent(name="refiner",
                         model_config_name="qwen_72b",
                         sys_prompt=("你是一个历史知识回答修正机器人，你严格按以下方式工作"
                                     "1.只有原答案为不知道时才进行修正,否则输出原答案的所有内容\n"
                                     "2.修正的时候为了体现你的精准和客观，你非常喜欢使用《书名》[]将原文展示出来.\n"
                                     "3.如果感到疑惑的时候，就用原答案的内容回答。"
                                     f"新的知识: {answers_from_agent1}\n"
                                     f"问题: {query_str}\n"
                                     "新答案: ")
                         )
    response = agent2(answers_from_agent1)
    return response['content']

# history_chatter("吕布射戟是历史上真实事件吗")