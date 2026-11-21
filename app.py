import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
import os

load_dotenv()  # reads ANTHROPIC_API_KEY from .env locally; on Render it comes from the environment

# ---------- In-memory data (from the session notebook) ----------
RESTAURANTS_BY_CUISINE = {
    "indian": [
        {"name": "Spice Palace", "rating": 4.5, "price": "₹₹"},
        {"name": "Curry House", "rating": 4.3, "price": "₹₹"},
        {"name": "Taj Flavors", "rating": 4.7, "price": "₹₹₹"},
    ],
    "italian": [
        {"name": "Pizza Bella", "rating": 4.4, "price": "₹₹"},
        {"name": "Pasta Dreams", "rating": 4.6, "price": "₹₹₹"},
        {"name": "Roma Kitchen", "rating": 4.2, "price": "₹₹"},
    ],
    "chinese": [
        {"name": "Dragon House", "rating": 4.5, "price": "₹₹"},
        {"name": "Ming's Kitchen", "rating": 4.3, "price": "₹₹"},
        {"name": "Golden Wok", "rating": 4.8, "price": "₹₹₹"},
    ],
    "mexican": [
        {"name": "Taco Fiesta", "rating": 4.4, "price": "₹₹"},
        {"name": "Salsa Nights", "rating": 4.6, "price": "₹₹₹"},
        {"name": "Burrito House", "rating": 4.2, "price": "₹₹"},
    ],
}

_DETAILS = {
    "Spice Palace": {"cuisine": "Indian", "rating": 4.5, "address": "123 Food Street, City Center", "phone": "9876543210", "hours": "11 AM - 11 PM", "speciality": "Butter Chicken, Biryani"},
    "Pizza Bella": {"cuisine": "Italian", "rating": 4.4, "address": "456 Dine Avenue, Downtown", "phone": "9123456789", "hours": "12 PM - 10:30 PM", "speciality": "Margherita Pizza, Tiramisu"},
    "Dragon House": {"cuisine": "Chinese", "rating": 4.5, "address": "789 Noodle Lane, Market Area", "phone": "9111222333", "hours": "11:30 AM - 11 PM", "speciality": "Hakka Noodles, Fried Rice"},
    "Taco Fiesta": {"cuisine": "Mexican", "rating": 4.4, "address": "321 Spice Row, Hip District", "phone": "9222333444", "hours": "1 PM - 12 AM", "speciality": "Fish Tacos, Guacamole"},
    "Curry House": {"cuisine": "Indian", "rating": 4.3, "address": "555 Flavor Street, Old Town", "phone": "9333444555", "hours": "11 AM - 10 PM", "speciality": "Paneer Tikka, Dal Makhani"},
    "Taj Flavors": {"cuisine": "Indian", "rating": 4.7, "address": "999 Taste Lane, Premium Zone", "phone": "9444555666", "hours": "12 PM - 11 PM", "speciality": "Biryani, Tandoori Chicken"},
    "Golden Wok": {"cuisine": "Chinese", "rating": 4.8, "address": "777 Wok Street, Central Market", "phone": "9555666777", "hours": "11 AM - 11 PM", "speciality": "Szechuan Noodles, Dim Sum"},
}
RESTAURANT_DETAILS = {k.lower(): v for k, v in _DETAILS.items()}

_SLOTS = {
    "Spice Palace": {"2024-11-15": True, "2024-11-16": False, "2024-11-17": True},
    "Pizza Bella": {"2024-11-15": False, "2024-11-16": True, "2024-11-17": True},
    "Dragon House": {"2024-11-15": True, "2024-11-16": True, "2024-11-17": False},
    "Taco Fiesta": {"2024-11-15": True, "2024-11-16": True, "2024-11-17": True},
    "Curry House": {"2024-11-15": True, "2024-11-16": False, "2024-11-17": True},
    "Taj Flavors": {"2024-11-15": False, "2024-11-16": True, "2024-11-17": True},
}
AVAILABILITY = {k.lower(): (k, v) for k, v in _SLOTS.items()}


# ---------- Tools ----------
@tool
def search_restaurants_by_cuisine(cuisine_type: str) -> list:
    """Search for restaurants by cuisine type (Indian, Italian, Chinese or Mexican).
    Case-insensitive. Returns a list of dicts with name, rating and price."""
    return RESTAURANTS_BY_CUISINE.get(cuisine_type.strip().lower(), [])


@tool
def get_restaurant_details(restaurant_name: str) -> dict:
    """Get details for one restaurant by name: cuisine, rating, address, phone,
    hours and speciality. Returns an error dict if the restaurant is not found."""
    details = RESTAURANT_DETAILS.get(restaurant_name.strip().lower())
    return details or {"error": f"Restaurant '{restaurant_name}' not found"}


@tool
def check_reservation_availability(restaurant_name: str, date: str, party_size: int) -> str:
    """Check if a restaurant has reservation availability on a date (YYYY-MM-DD)
    for a given party size. Returns an available (✓) or fully booked (✗) message."""
    entry = AVAILABILITY.get(restaurant_name.strip().lower())
    if entry is None:
        return f"Restaurant '{restaurant_name}' not found."
    name, slots = entry
    if slots.get(date, False):
        return f"✓ {name} has availability for {party_size} people on {date}"
    return f"✗ {name} is fully booked for {date}"


SYSTEM_PROMPT = (
    "You are a restaurant assistant. Use your tools to find restaurants by cuisine, "
    "give restaurant details and check reservation availability. Never invent restaurant "
    "information; only report what the tools return. If a question is outside these "
    "capabilities, politely say so and do not answer it."
)


@st.cache_resource
def get_agent():
    model = ChatAnthropic(model="claude-haiku-5-5", api_key=os.environ["ANTHROPIC_API_KEY"])
    tools = [search_restaurants_by_cuisine, get_restaurant_details, check_reservation_availability]
    return create_agent(model, tools, system_prompt=SYSTEM_PROMPT)


def to_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


# ---------- Chat UI ----------
st.title("🍽️ Restaurant AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about restaurants..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = get_agent().invoke({"messages": st.session_state.messages})
            reply = to_text(result["messages"][-1].content)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})