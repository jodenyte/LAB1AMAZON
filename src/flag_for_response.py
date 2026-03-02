"""Flag reviews that warrant a response."""


def flag_for_response(
    text: str,
    sentiment: str,
    length_threshold: int = 120,
    keywords: list[str] | None = None,
) -> bool:
    """
    Determine whether a review should be flagged for human response.

    Args:
        text: The review text.
        sentiment: Predicted or ground-truth sentiment ('positive', 'neutral', 'negative').
        length_threshold: Flag short reviews below this character count (default 120).
        keywords: Optional list of words/phrases that trigger a flag (case-insensitive).

    Returns:
        True if the review should be flagged for response, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False

    text_lower = text.lower().strip()
    is_short = len(text_lower) < length_threshold
    is_negative = sentiment and sentiment.lower() == "negative"

    # Flag negative reviews, especially short ones
    if is_negative and is_short:
        return True

    # Flag if any keyword appears
    if keywords:
        for kw in keywords:
            if kw and kw.lower() in text_lower:
                return True

    return False
