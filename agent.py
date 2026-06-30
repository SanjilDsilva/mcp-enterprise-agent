import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 1. The actual Python function (our "Database")
def get_hotel_price(city: str):
    print(f"\n[SYSTEM LOG: Executing get_hotel_price for {city}...]")
    prices = {"london": "£250", "paris": "£300", "tokyo": "£150", "kochi": "£50"}
    return prices.get(city.lower(), "Price data not found.")

def get_weather(city: str):
    print(f"\n[SYSTEM LOG: Checking weather for {city}...]")
    weather = {"london": "Rainy, 12°C", "paris": "Sunny, 20°C", "tokyo": "Cloudy, 18°C", "kochi": "Humid, 32°C"}
    return weather.get(city.lower(), "Weather data not found.")

# 2. The Tool Definition (The menu we hand to the LLM)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_hotel_price",
            "description": "Get the current nightly hotel room price for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city, e.g., London, Tokyo, Kochi",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the temperature for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city, e.g., London, Tokyo, Kochi",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

messages = [{"role": "system", "content": "You are a concise, highly skilled Enterprise AI Agent."}
]

while True:
    user_input = input("You: ")

    if user_input.lower() in ['quit', 'exit']:
        break
    messages.append({"role": "user", "content": user_input})

    chat_completion = client.chat.completions.create(
        messages = messages,
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=500,
        tools=tools,
        tool_choice="auto"
    )

    # 1. Get the full message object from the AI
    response_message = chat_completion.choices[0].message

    # 2. Check if the LLM wants to use a tool
    if response_message.tool_calls:
        print(f"\n[AGENT THOUGHT: I need to use a tool!]")
        
        # We MUST append the LLM's tool request to memory first so it remembers asking
        messages.append(response_message)
        
        # 3. Loop through the tools the LLM wants to call
        for tool_call in response_message.tool_calls:
            import json
            args = json.loads(tool_call.function.arguments)
            city_name = args.get("city")
            
            tool_output = "" # Create a generic variable to hold the answer
            
            # Route to the correct tool!
            if tool_call.function.name == "get_hotel_price":
                tool_output = get_hotel_price(city_name)
            elif tool_call.function.name == "get_weather":
                tool_output = get_weather(city_name)
                
            # 4. Pass the result BACK to the LLM's memory
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": str(tool_output) # Send whatever the tool_output was
            })
        
        # 5. Make a SECOND API call now that the LLM has the database info
        second_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile"
        )
        final_response = second_completion.choices[0].message.content
        print(f"Agent: {final_response}")
        messages.append({"role": "assistant", "content": final_response})
        
    else:
        # The LLM didn't need a tool, just print its normal text response
        ai_response = response_message.content
        print(f"Agent: {ai_response}")
        messages.append({"role": "assistant", "content": ai_response})