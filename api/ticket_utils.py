from typing import List, Dict, Any
import requests

class TicketInfo:
    def __init__(self, ticket_id: int, event_name: str, date: str, price: float, availability: str):
        self.ticket_id = ticket_id
        self.event_name = event_name
        self.date = date
        self.price = price
        self.availability = availability

def fetch_ticket_information(event_name: str) -> List[TicketInfo]:
    api_url = "https://app.ticketmaster.com/discovery/v2/events"
    api_key = ""
    params = {"keyword": event_name, "apikey": api_key}
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        ticket_data = response.json()
        
        tickets = []
        for event in ticket_data['_embedded']['events']:
            tickets.append(TicketInfo(
                ticket_id=event['id'],
                event_name=event['name'],
                date=event['dates']['start']['localDate'],
                price=event['priceRanges'][0]['min'] if 'priceRanges' in event else 0.0,
                availability=event['dates']['status']['code']
            ))
        return tickets
    except requests.RequestException as e:
        print(f"Error fetching ticket information: {e}")
        return []


def get_ticket_details(event_name: str) -> Dict[str, Any]:
    tickets = fetch_ticket_information(event_name)
    if not tickets:
        return {"message": "No tickets found for the specified event."}
    
    return {
        "event_name": event_name,
        "tickets": [ticket.__dict__ for ticket in tickets]
    }