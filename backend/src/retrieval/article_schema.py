from pydantic import BaseModel
from langchain_classic.chains.query_constructor.base import AttributeInfo

class RetrievedChunk(BaseModel):
    text: str
    score: float | None = None
    article_number: int
    # reference_url: str | None = None


metadata_field_info = [
    AttributeInfo(
        name="article_number",
        description="The article number in the law (ماده). For example 1, 2, 105.",
        type="integer",
    ),
    AttributeInfo(
        name="title",
        description="The full Persian title of the article, e.g. 'ماده 1' or 'ماده 105'.",
        type="string",
    ),
    AttributeInfo(
        name="part_number",
        description="The part (باب) number this article belongs to. For example 1 for باب اول.",
        type="integer",
    ),
    AttributeInfo(
        name="chapter_number",
        description="The chapter (فصل) number this article belongs to within its part. For example 1 for فصل اول.",
        type="integer",
    ),
    AttributeInfo(
        name="article_id",
        description="The unique internal ID of the article on the source website, e.g. 'txs5000107875'.",
        type="string",
    ),
    AttributeInfo(
        name="chunk_index",
        description="Index of this chunk within the article, starting from 0. Articles with a lot of text may be split into multiple chunks.",
        type="integer",
    ),
    AttributeInfo(
        name="reference_url",
        description="The direct URL to this article on the official Iranian tax law website.",
        type="string",
    ),
]

document_content_description = """
Articles from the Iranian Direct Taxes Law (قانون مالیات‌های مستقیم), including the main article body and any associated notes (تبصره).
Covers topics such as taxable persons, income tax, corporate tax, exemptions, tax rates, and procedural rules.
""".strip()