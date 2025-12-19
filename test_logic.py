def check_logic(user_message):
    tech_words = ["how", "explain", "activate", "connect", "step", "tutorial", "manual", "guide", "help", "camera", "ip", "monitor", "nvr"]
    u_msg_lower = user_message.lower()
    
    triggers = any(w in u_msg_lower for w in tech_words) or len(user_message.split()) > 3
    return triggers

test_cases = [
    "explain how to",       # Should True
    "Explain how to...",    # Should True (Cap)
    "Hello",                # Should False
    "activateme",           # Should True (substring)
    "How do I...",          # Should True
    "Simple",               # Should False
    "Short text",           # Should False
    "A very long sentence about nothing related to tech" # Should True (len > 3)
]

print("Running Logic Test:")
for case in test_cases:
    res = check_logic(case)
    print(f"'{case}': {res}")
