from langchain_community.embeddings import DashScopeEmbeddings


def embedding_method():
    embedding = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key="sk-a55c969f708b43429ec601d536f9efac")
    return embedding



def book_name(path):
    _mapping = {}
    _mapping["baihuabeiqishu"] = "北齐书"
    _mapping["baihuabeishi.txt"] = "北史"
    _mapping["baihuachenshu.txt"] = "陈书"
    _mapping["baihuahanshu.txt"] = "汉书"
    _mapping["baihuahouhanshu.txt"] = "后汉书"
    _mapping["baihuajinshi.txt"] = "金史"
    _mapping["baihuajinshu.txt"] = "晋书"
    _mapping["baihuajiutangshu.txt"] = "旧唐书"
    _mapping["baihuajiuwudaishi.txt"] = "旧五代史"
    _mapping["baihualiangshu.txt"] = "梁书"
    _mapping["baihualiaoshi.txt"] = "辽史"
    _mapping["baihuamingshi.txt"] = "明史"
    _mapping["baihuananqishu.txt"] = "南齐书"
    _mapping["baihuananshi.txt"] = "南史"
    _mapping["baihuasanguozhi.txt"] = "三国志"
    _mapping["baihuashiji.txt"] = "史记"
    _mapping["baihuasongshi.txt"] = "宋史"
    _mapping["baihuasongshu.txt"] = "宋书"
    _mapping["baihuasuishu.txt"] = "隋史"
    _mapping["baihuaweishu.txt"] = "魏书"
    _mapping["baihuaxintangshi.txt"] = "新唐史"
    _mapping["baihuaxinwudaishi.txt"] = "新五代史"
    _mapping["baihuayuanshi.txt"] = "元史"
    _mapping["baihuazhoushu.txt"] = "周书"
    _mapping["sanguozhi.txt"] = "三国志"

    for name in _mapping:
        if name in path:
            return _mapping[name]
    return "未名"
