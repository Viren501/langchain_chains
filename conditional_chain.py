from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables.base import RunnableParallel, RunnableLambda # To execute chains in parallel
# RunnableLambda convert a lambda function to runnable and it can be used as chain
from langchain_core.runnables.branch import RunnableBranch # To branch chains using if-else
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-26B-A4B-it",
    task="text-generation"
) 

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1= PromptTemplate(
    template='Classfy the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables= {'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# To check the classifier_chain working
# result = classifier_chain.invoke({'feedback': 'This is a wonderful smartphone'}).sentiment
# print(result)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),#(condition1, chain1),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),#(condition2, chain2),
    RunnableLambda(lambda x: "could not find sentiment")#default chain
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()
