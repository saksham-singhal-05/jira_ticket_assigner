# jira_ticket_assigner

## Prerequisites
- database named `tickets`
- `issues` table in tickets

**Also** add a column for 'embeddings' in  `issues`
>>ALTER TABLE issues ADD COLUMN embedding float8[];


## How to use:
- Edit config.yaml to devices postgresql configurations
- If you want to use OpenAi embeddings add OpenAi key and change the first line of config.yaml
- Run daily training files once a day(setup files: 01_embed_new_tickets.pyand 02_train_model.py)
- run streamlit app for each ticket/assignment.py to save in db
