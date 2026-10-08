def select_chunks(chunks, max_chapter, max_words):
    selected = []
    seen = set()
    used = 0

    ordered = sorted(
        enumerate(chunks),
        key=lambda item: (-item[1]['score'], item[0])
    )

    for _, chunk in ordered:
        if chunk['chapter'] > max_chapter:
            continue
        if chunk['id'] in seen:
            continue

        words = len(chunk['text'].split())
        if used + words > max_words:
            continue

        selected.append(chunk)
        seen.add(chunk['id'])
        used += words

    return selected
