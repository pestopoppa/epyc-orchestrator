from src.escalation import should_skip_cheap_first
if should_skip_cheap_first(request.prompt, routing.difficulty_band):
    return None