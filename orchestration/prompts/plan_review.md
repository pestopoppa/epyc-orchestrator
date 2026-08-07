Review plan. Reply JSON ONLY:
{{"d":"ok|reorder|drop|add|reroute","s":0.0-1.0,"f":"<15 words","p":[]}}

d=decision, s=confidence, f=feedback, p=patches (optional)
drop=discard this plan and continue with the normal no-plan route; never refuse the task
Patch format: {{"step":"S1","op":"reroute|drop|add|reorder","v":"new_value"}}

Task: {objective}
Type: {task_type}
Plan:
{steps_section}

Verdict:
