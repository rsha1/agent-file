from letta_client import Letta
from composio import Action, ComposioToolSet, App


client = Letta(base_url = "http://localhost:8283")

# add github to composio toolset
toolset = ComposioToolSet()
request = toolset.initiate_connection(app=App.GITHUB)
print(f"Open this URL to authenticate: {request.redirectUrl}")

# create composio tool 
github_star_tool = client.tools.add_composio_tool(
    composio_action_name=Action.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER.name
)

print(github_star_tool)

# create agent
agent = client.agents.create(
    name="composio_github_star_agent", 
    description="Agent that stars repos on github",
    tool_ids=[github_star_tool.id],
    memory_blocks=[
        {
            "label": "human", 
            "value": "Name: Sarah"
        }, 
        {
            "label": "persona", 
            "value": "You are an agent helping the human star cool repos!"
        }
    ], 
    model="openai/gpt-4o-mini", 
    embedding="openai/text-embedding-ada-002"
)

print(agent.id)
print("tools", [t.name for t in agent.tools])




