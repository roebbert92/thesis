def dataset_similarity(first_json_path: str, second_json_path: str):
    # build database (gazetteers (entities), sentences) if not exists for each path
    # database is cosine similarity
    # if both paths are the same, filter out same doc.id
    # Optional:
    # for gazetteers: 0.5*cosine(content) + 0.5 * KL(1 if same type, 0 if other type)
    # for sentences: 0.5*cosine(content) + 0.5 * Avg(Gazetteers)
    # take first score

    # avg

    # repeat for other side if not the same path
    return {"entities": (0.0, 0.0), "sentences": (0.0, 0.0)}