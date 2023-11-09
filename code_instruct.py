


# To use RAG, you need to install the following packages:
# pip install "pyautogen[retrievechat]"
# pip install "unstructured[all-docs]"

import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent



local_config_list = [
        {
            'model': 'WizardCoder-Python-34B-V1.0',
            'api_key': 'any string here is fine',
            'api_type': 'openai',
            'api_base': "http://localhost:1234/v1",
        }
]

gpt_config_list = [
    {
        'model': 'gpt-3.5-turbo', # 'gpt-4',
        'api_key': 'YOUR-API-KEY',
    }
]


rag_proxy = RetrieveUserProxyAgent(
    name="rag_proxy",
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
    },
    code_execution_config=False,  
)

planner = autogen.AssistantAgent(
    name="planner",
    llm_config={"config_list": gpt_config_list},
    # the default system message of the AssistantAgent is overwritten here
    system_message='''
    You are a dedicated AI for data science and machine learning tasks. Your role is to suggest algorithmic approaches, data preprocessing steps, and statistical methods to another AI assistant working on a specific task. 
    Avoid providing concrete code, but instead, offer high-level methodologies and the reasoning behind them. For actions not inherently related to coding or algorithmic reasoning, convert them into steps that can be implemented algorithmically. 
    For instance, building ML models could be translated to writing code that acquires, processes, and structures the required data, followed by selecting, training, and evaluating appropriate algorithms. 
    Once a task is executed, inspect the results critically. Evaluate the performance using appropriate metrics, and suggest optimizations or alternative approaches if the results are not satisfactory. 
    If there's an error in execution, help in diagnosing the error and propose potential fixes. Your goal is to guide the other AI towards achieving the best possible outcome on the task at hand.
    '''
)

planner_user = autogen.UserProxyAgent(
    name="planner_user",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER", # ALWAYS, NEVER, or TERMINATE
    llm_config={"config_list": gpt_config_list},
)

def ask_planner(message):
    planner_user.initiate_chat(planner, message=message)
    # return the last message received from the planner
    return planner_user.last_message()["content"]
    
def retrieve_content(message, n_results=3):
        rag_proxy.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = rag_proxy._check_update_context(message)
        if (update_context_case1 or update_context_case2) and rag_proxy.update_context:
            rag_proxy.problem = message if not hasattr(rag_proxy, "problem") else rag_proxy.problem
            _, ret_msg = rag_proxy._generate_retrieve_user_reply(message)
        else:
            ret_msg = rag_proxy.generate_init_message(message, n_results=n_results)
        return ret_msg if ret_msg else message


# create an AssistantAgent named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "seed": 42,  # seed for caching and reproducibility
        "config_list": gpt_config_list,  # a list of OpenAI API configurations
        # "temperature": 0,  # temperature for sampling | # always control from LM Studio
        # "repeat_penalty": 1.1,  # penalty to reduce the chance of repeating the same text | # always control from LM Studio
        # "stop": ["\n\n\n\n"],  # tokens at which text generation is stopped
        "retry_wait_time": 5,  # time interval to wait before retrying a failed request | Handle Rate Limit Error
        "max_retry_period": 60,  # total timeout allowed for retrying failed requests | Handle Rate Limit Error
        "request_timeout": 200,  # timeout for a single request | Handle timeout Error
            "functions": [
            {
                "name": "ask_planner",
                "description": "ask planner to: 1. get a plan for finishing a task, 2. verify the execution result of the plan and potentially suggest new plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "question to ask planner. Make sure the question include enough context, such as the code and the execution result. The planner does not know the conversation between you and the user, unless you share the conversation with the planner.",
                        },
                    },
                    "required": ["message"],
                },
            }, {
            "name": "retrieve_content",
            "description": "retrieve content for code generation and question answering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                        }
                    },
                "required": ["message"],
                },
            },
        
        ],
    }, 
    function_map={"ask_planner": ask_planner}, # Added  
    system_message='''
                    You are responsible for executing a task at hand and writing the necessary full code to complete it. 
                    You can ask user_proxy to execute the code you write, and ask planner for help in planning the task and evaluating the results.
                    Run a test on a subset of the original data to ensure the code is working as expected.
                    '''
)


# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir":"_output", "use_docker":False}, # If you have problems with agents running pip install or get errors similar to Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'), you can choose 'python:3' as image as shown in the code example above and that should solve the problem.
    # Also make sure docker is running and python:3 container is running (even for native)
    llm_config={"config_list": gpt_config_list},
    function_map={"ask_planner": ask_planner},
    system_message=""""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""" #  All code must be written and tested for full satisfaction. 
)

# assistant.reset()


user_proxy.initiate_chat(
    assistant,
    clear_history=True, # to continue a finished conversation
    message="""write python script create a list of numbers from 1 to 100, and print all the numbers that are divisible by 3 and 5 and save the script to test.py"""
)
