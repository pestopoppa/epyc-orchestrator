Reformat the following answer to strictly satisfy this format constraint: {format_spec}

Original question: {prompt}

Original answer:
{answer}

Compliance rules (apply in order, verbatim):
1. Treat the format constraint as absolute: exact JSON structure, field names, field order, key casing, list style, delimiters, and wrapping must match the spec character-for-character where specified.
2. If the constraint forbids markdown/code fences, explanations, or preamble, emit none — not even a leading or trailing blank line beyond what the spec allows.
3. If the constraint requires code only, output raw code starting at the first token (e.g., an import or definition) with no fences and no commentary.
4. Preserve the substantive content of the original answer; change only its formatting to satisfy the constraint. Do not add, summarize, or editorialize.
5. Before emitting, silently verify the candidate output against every element of the constraint (schema, ordering, casing, count, length limits). If any element fails, re-emit corrected output. Only the final, fully compliant text is returned.

Output ONLY the reformatted answer. Do not add explanations, preamble, or any text outside the required format.