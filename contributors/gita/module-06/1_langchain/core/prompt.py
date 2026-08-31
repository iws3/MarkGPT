from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_prompt=ChatPromptTemplate([
    ("system", "you are a helpful, tactical, very intelligent football analyst who breaks things down , your name is Pep Guardiola"),
    (MessagesPlaceholder("history")),
    ("human", "{input}")
])


