import json
import random

class RecommendationManager:
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.last_shown = {}   

    def get_next_recommendation(self, stress_level):
        options = self.data.get(stress_level, [])
        if not options:
            return None
        
        #  last shown type 
        last_type = self.last_shown.get(stress_level)
        filtered = [rec for rec in options if rec["type"] != last_type]
        
        if not filtered:   
            filtered = options
        
        choice = random.choice(filtered)
        self.last_shown[stress_level] = choice["type"]
        return choice

    def adapt_recommendation(self, stress_level, user_response):

        if user_response.lower() in ["no", "n"]:
            self.last_shown[stress_level] = None
