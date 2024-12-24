from pymongo import MongoClient
import streamlit as st
import openai


try:
    connection_string = st.secrets["CONNECTION_STRING"]
    print(connection_string)
except KeyError:
    st.error("CONNECTION_STRING not found in secrets file.") 
    

try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    print(connection_string)
except KeyError:
    st.error("CONNECTION_STRING not found in secrets file.") 
    
    
client = MongoClient(connection_string)
db = client['equipt_master']

def get_collection_stats_and_schema():
    collections = db.list_collection_names(filter={"name": {"$in": ["serializedAsset", "productCategory", "invoices", "products", "warehouses", "customerAccounts", "supplierAccounts", "brands"]}})
    all_stats = {}

    for collection_name in collections:
        stats = db.command("collStats", collection_name)
        # print(f"stats for collection {collection_name}")
        stats = db.command("collStats", collection_name)
        # print(f"stats {stats}")
        sample = db[collection_name].find(limit=1000)
        # print(f"sample {sample}")

        schema = {field: type(value).__name__ for doc in sample for field, value in doc.items()} if sample else {} 
        
        all_stats[collection_name] = {
            "stats": stats,
            "schema": schema
        }
    return all_stats



all_stats = get_collection_stats_and_schema()

def format_openai_prompt_for_multiple_collections(all_stats):
    prompt = "I have the following MongoDB collections with their stats and schemas:\n\n"

    for collection, details in all_stats.items():
        prompt += f"Collection: {collection}\n"
        prompt += f"Stats: {details['stats']}\n"
        prompt += f"Schema: {details['schema']}\n\n"

    prompt += """Given these collections, create a schema-aware Retrieval-Augmented Generation (RAG) system that:
    1. Retrieves data from multiple collections.
    2. Understands relationships between collections.
    3. Returns precise answers based on the query context."""
    return prompt


prompt = format_openai_prompt_for_multiple_collections(all_stats)

# print(prompt)

def query_openai(prompt):

  keywords = prompt.lower().split() 

  relevant_collections = []
  relevant_fields = {}
  for collection, details in all_stats.items():
    schema = details.get("schema", {}) 
    for field, field_type in schema.items():
      if any(keyword in field.lower() for keyword in keywords):
        relevant_collections.append(collection)
        relevant_fields.setdefault(collection, []).append(field)

  message = f"""You are tasked with analyzing MongoDB data and answering user queries.

  **User Query:** {prompt}

  **Keywords:** {', '.join(keywords)}  # List extracted keywords

  **Relevant Collections:**
  {', '.join(relevant_collections)}

  **Relevant Fields (if any):**
  {', '.join([f"{c}: {', '.join(f)}" for c, f in relevant_fields.items()])}

  Please analyze the data and provide the most relevant information in response to the user's query. Ensure the response is structured and easy to understand. do all the required calculation necessary. do not provide additional information which is not necessary. If the required data is not available in the provided collections., clearly state 'Required data not available'.
  """

  # Send the message to OpenAI and return the response
  response = openai.chat.completions.create(
      model="text-davinci-003",
      max_completion_tokens=300,
      max_tokens=4000,
      messages=[
          {"role": "system", "content": message},
          {"role": "user", "content": prompt},
      ]
  )
  return response.choices[0].message.content 


def main():
    st.title("Multi-Collection MongoDB RAG System")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! How can I assist you with the Vebholic Private Limited database?",
            }
        ]

    st.sidebar.title("MongoDB Connection Info")
    with st.sidebar.form("mongodb_form"):
        mongo_uri = st.text_input("Mongo URI", value=connection_string)
        db_name = st.text_input("Database Name", value=db)
        if st.form_submit_button("Connect"):
            st.session_state.mongo_uri = mongo_uri
            st.session_state.db_name = db_name

    user_input = st.chat_input("Ask your query...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            all_stats = get_collection_stats_and_schema()

            prompt = format_openai_prompt_for_multiple_collections(
                all_stats
            )

            response = query_openai(prompt)

            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])



if __name__ == "__main__":
    main()
