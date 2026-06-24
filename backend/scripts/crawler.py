from bs4 import BeautifulSoup
import json
import re

def parse_law_text(element):
    refs = []
    
    # Extract cross-references before stripping
    for a in element.find_all("a", href=True):
        href = a.get("href", "")
        if "lwvi" in href:
            refs.append({
                "text": a.get_text(strip=True),
                "href": "https://regulation.tax.gov.ir/" + href
            })
    
    return {
        "text": element.get_text(separator="\n", strip=True),
        "cross_references": refs
    }

def parse_article(soup):
    articles = []
    part_num = 0
    chapter_num = 0
    for row in soup.select("div.row-itm"):
        is_note=False
        is_section=False

        row_id = row.get("id", "").replace("row_", "")
        
        title_el = row.select_one("a.row-ti")
        title = title_el.get_text(strip=True) if title_el else ""

        num_match = re.search(r"\d+", title)
        div_num = int(num_match.group()) if num_match else None
                
        if "باب" in title:
            part_num += 1
            chapter_num = 0
            continue
        elif "فصل" in title:
            chapter_num += 1
            continue
        elif "تبصره" in title:
            is_note = True
        elif "ماده" not in title:
            is_section = True
        else: 
            article_num = div_num
        
        # Main content div
        content_el = row.select_one(f"div#content_{row_id} div.lwe-text-view")
        if not content_el:
            continue
        
        parsed = parse_law_text(content_el)
        
        if is_note:
            last_article = articles.pop()
            last_article["text"] = last_article["text"] + "\n" + parsed["text"]
            last_article["notes"].append(parsed["text"])
            last_article["cross_references"] += parsed["cross_references"]
            articles.append(last_article)
        elif is_section:
            last_article = articles.pop()
            last_article["text"] = last_article["text"] + "\n" + parsed["text"]
            last_article["sections"].append(parsed["text"])
            last_article["cross_references"] += parsed["cross_references"]
            articles.append(last_article)
        else:
            if chapter_num == 0:
                chapter_num += 1
            articles.append({
                "id": row_id,
                "article_number": article_num,
                "title": title,
                "text": parsed["text"],
                "part_number": part_num,
                "chapter_number": chapter_num,
                "reference_url": "https://regulation.tax.gov.ir/lwvi?ty=qh&lid=txs5000060569&pid=" + row_id,
                "notes": [],
                "sections": [],
                "cross_references": parsed["cross_references"],
            })
    
    return articles


with open("../data/raw/dl.html", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

articles = parse_article(soup)

with open("../data/raw/dtl.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)