"""Truncation must fit the model budget without moving option markers."""

from lakun.encoding import MAX_CONTEXT, VISUAL_TOKENS, encode_question


class SpaceTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    mask_token = "[MASK]"

    def __init__(self):
        self.vocabulary = {}

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        assert not add_special_tokens
        ids = []
        for word in text.split():
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary) + 10
            ids.append(self.vocabulary[word])
        return {"input_ids": ids[:max_length] if truncation else ids}


def test_long_state_is_truncated_for_text_and_image():
    tokenizer = SpaceTokenizer()
    row = {"id": "test:0", "type": "choice", "question": "What happened?",
           "criteria": ["refund", "other"]}
    long_state = "start " + "details " * 2000 + "ending"
    text_ids, text_markers = encode_question(tokenizer, long_state, row, image=False)
    image_ids, image_markers = encode_question(tokenizer, long_state, row, image=True)
    assert len(text_ids) == MAX_CONTEXT
    assert len(image_ids) == MAX_CONTEXT - VISUAL_TOKENS
    assert text_markers == image_markers
    assert all(text_ids[index] == tokenizer.mask_token_id for index in text_markers)
    assert text_ids[-1] == image_ids[-1] == tokenizer.sep_token_id
    assert tokenizer.vocabulary["start"] in text_ids and tokenizer.vocabulary["ending"] not in text_ids


def test_oversized_question_and_twenty_options_fit_head():
    tokenizer = SpaceTokenizer()
    options = [f"option{i} " + "description " * 100 for i in range(20)]
    row = {"id": "test:1", "type": "choice", "question": "question " * 500,
           "criteria": options}
    ids, markers = encode_question(tokenizer, "state " * 1000, row, image=True)
    assert len(ids) == MAX_CONTEXT - VISUAL_TOKENS
    assert len(markers) == len(options)
    assert markers == sorted(markers)
    assert all(ids[index] == tokenizer.mask_token_id for index in markers)


def test_short_input_keeps_complete_state_and_options():
    tokenizer = SpaceTokenizer()
    row = {"id": "test:2", "type": "choice", "question": "What happened?",
           "criteria": ["refund", "other"]}
    ids, markers = encode_question(tokenizer, "short state", row, image=False)
    assert len(markers) == 2
    assert ids[-3:-1] == [tokenizer.vocabulary["short"], tokenizer.vocabulary["state"]]
