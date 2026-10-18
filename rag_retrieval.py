"""Hybrid retrieval and section context for existing sequential MinerU chunks.

Works with the current index: no deletion or new embeddings are required.
"""
import math
import re
from collections import Counter


def _tokens(text):
    text = re.sub(r'<[^>]*>', ' ', text).casefold()
    tokens = []
    stop = {'公司', '什么', '哪些', '多少', '如何', '是否', '请问', '情况'}
    for part in re.findall(r'[a-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+', text):
        if re.fullmatch(r'[\u4e00-\u9fff]+', part):
            tokens.extend(part[i:i+2] for i in range(len(part)-1) if part[i:i+2] not in stop)
        else:
            tokens.append(part)
    return tokens


def _normalized(text):
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', text.casefold())


def _phrase(question):
    question = _normalized(question)
    for prefix in ('请问', '请介绍', '介绍一下', '告诉我'):
        if question.startswith(prefix):
            question = question[len(prefix):]
    for suffix in ('是什么', '有哪些', '是多少', '怎么样'):
        if question.endswith(suffix):
            question = question[:-len(suffix)]
    return question


def _heading_only(text):
    text = text.strip()
    return text.startswith('## ') and len(text) <= 100 and not re.search(r'[。；;]|<table', text)


def _heading_rank(text):
    text = text.lstrip('# ').strip()
    if re.match(r'第[一二三四五六七八九十\d]+[章节]', text):
        return 0
    if re.match(r'[一二三四五六七八九十]+[、．.]', text):
        return 1
    if re.match(r'[（(]\d+[）)]', text):
        return 3
    return 2


def _heading_name(text):
    return _normalized(re.sub(r'^[#\s]*[一二三四五六七八九十\d]+[、．.]', '', text))


def _chunk_position(identifier):
    prefix, marker, number = identifier.rpartition('_chunk_')
    return (prefix, int(number)) if marker and number.isdigit() else None


def _table_signature(text):
    return text.split('</tr>', 1)[0] if '<table' in text else None


def _context(anchor, records, max_chars=2600, exclude_bodies=None):
    seed = records[anchor]
    position = _chunk_position(anchor)
    if position is None:
        return seed['text'], [anchor], dict(seed['metadata'])
    prefix, number = position
    source = seed['metadata'].get('source')

    def neighbor(index):
        identifier = f'{prefix}_chunk_{index}'
        item = records.get(identifier)
        return identifier if item and item['metadata'].get('source') == source else None

    included = [anchor]
    seed_is_heading = _heading_only(seed['text'])
    table_signature = _table_signature(seed['text'])
    chars = len(seed['text'])
    nearest_heading = seed['text'] if seed['text'].lstrip().startswith('## ') else None
    heading_seen = nearest_heading is not None
    # A retrieved table row often belongs to a larger table. Include earlier
    # siblings with the same header and page, and their immediately preceding title.
    for index in range(number-1, max(-1, number-5), -1):
        identifier = neighbor(index)
        if identifier is None:
            break
        item = records[identifier]
        if _heading_only(item['text']):
            if nearest_heading is not None and _heading_rank(item['text']) >= _heading_rank(nearest_heading) and _heading_name(item['text']) != _heading_name(nearest_heading):
                break
            included.insert(0, identifier)
            chars += len(item['text'])
            nearest_heading = item['text']
            heading_seen = True
            continue
        if table_signature and not heading_seen and _table_signature(item['text']) == table_signature and item['metadata'].get('page_idx') == seed['metadata'].get('page_idx') and chars + len(item['text']) <= max_chars:
            included.insert(0, identifier)
            chars += len(item['text'])
            continue
        break

    has_body = not seed_is_heading
    for index in range(number+1, number+9):
        identifier = neighbor(index)
        if identifier is None:
            break
        item = records[identifier]
        text = item['text']
        if text.lstrip().startswith('## '):
            if has_body:
                break
            # Before the body, accept child or equivalent headings, but do not
            # cross into a different sibling section merely to find any text.
            if _heading_rank(text) <= _heading_rank(seed['text']) and _heading_name(text) != _heading_name(seed['text']):
                break
        elif table_signature and (_table_signature(text) != table_signature or item['metadata'].get('page_idx') != seed['metadata'].get('page_idx')):
            break
        if chars + len(text) > max_chars:
            break
        included.append(identifier)
        chars += len(text)
        has_body = has_body or not _heading_only(text)
    if exclude_bodies:
        included = [identifier for identifier in included if _heading_only(records[identifier]['text']) or identifier not in exclude_bodies]
    bodies = [identifier for identifier in included if not _heading_only(records[identifier]['text'])]
    metadata = dict(seed['metadata'])
    if 'page_idx' not in metadata:
        for identifier in included:
            if 'page_idx' in records[identifier]['metadata']:
                metadata['page_idx'] = records[identifier]['metadata']['page_idx']
                break
    if included:
        metadata['context_range'] = f'{_chunk_position(included[0])[1]}–{_chunk_position(included[-1])[1]}'
    return '\n\n'.join(records[identifier]['text'] for identifier in included), bodies, metadata


def _distance(query, embedding, metric):
    if metric == 'cosine':
        product = sum(float(a)*float(b) for a,b in zip(query, embedding))
        norm = math.sqrt(sum(float(a)**2 for a in query) * sum(float(b)**2 for b in embedding))
        return 1-product/norm if norm else 1.0
    if metric == 'ip':
        return 1-sum(float(a)*float(b) for a,b in zip(query, embedding))
    return float(sum((float(a)-float(b))**2 for a,b in zip(query, embedding)))


def retrieve_with_context(collection, question, query_embedding, n_results=5):
    empty = {key: [[]] for key in ('ids','documents','metadatas','distances')}
    count = collection.count()
    if not count:
        return empty
    limit = max(1, min(int(n_results), count))
    candidate_count = min(count, max(15, limit*3))
    dense = collection.query(query_embeddings=[query_embedding], n_results=candidate_count,
                             include=['documents','metadatas','distances'])
    stored = collection.get(include=['documents','metadatas'])
    records = {identifier: {'text': text, 'metadata': metadata or {}}
               for identifier,text,metadata in zip(stored['ids'],stored['documents'],stored['metadatas'])}
    dense_rank = {identifier: rank for rank,identifier in enumerate(dense['ids'][0], 1)}
    distances = {identifier: float(value) for identifier,value in zip(dense['ids'][0],dense['distances'][0])}
    # BM25 on Chinese bigrams and Latin/numeric terms complements vectors when
    # the question is a literal section title or financial metric.
    frequencies = {identifier: Counter(_tokens(item['text'])) for identifier,item in records.items()}
    document_frequency = Counter(token for terms in frequencies.values() for token in terms)
    average_length = sum(sum(terms.values()) for terms in frequencies.values()) / max(len(records),1)
    query_terms = set(_tokens(question))
    lexical = {}
    for identifier,terms in frequencies.items():
        length = sum(terms.values())
        score = 0.0
        for token in query_terms:
            tf = terms[token]
            if tf:
                idf = math.log(1 + (len(records)-document_frequency[token]+.5)/(document_frequency[token]+.5))
                score += idf * tf * 2.2 / (tf + 1.2*(.25 + .75*length/max(average_length,1)))
        if score:
            lexical[identifier] = score
    lexical_order = sorted(lexical, key=lexical.get, reverse=True)[:candidate_count]
    lexical_rank = {identifier: rank for rank,identifier in enumerate(lexical_order,1)}
    candidates = set(dense_rank) | set(lexical_rank)
    phrase = _phrase(question)

    def score(identifier):
        value = (1/(60+dense_rank[identifier]) if identifier in dense_rank else 0)
        value += (1/(60+lexical_rank[identifier]) if identifier in lexical_rank else 0)
        if len(phrase) >= 4 and phrase in _normalized(records[identifier]['text']):
            value += .1
        return value

    missing_distances = [identifier for identifier in candidates if identifier not in distances]
    if missing_distances:
        embeddings = collection.get(ids=missing_distances, include=['embeddings'])
        metric = (collection.metadata or {}).get('hnsw:space','l2')
        for identifier,embedding in zip(embeddings['ids'],embeddings['embeddings']):
            if embedding is not None:
                distances[identifier] = _distance(query_embedding, embedding, metric)
    ordered = sorted((identifier for identifier in candidates if identifier in records and identifier in distances),
                     key=lambda identifier: (-score(identifier), distances[identifier], identifier))
    result = {key: [[]] for key in empty}
    used_bodies = set()
    for identifier in ordered:
        text,bodies,metadata = _context(identifier,records,exclude_bodies=used_bodies)
        if not bodies:
            continue
        used_bodies.update(bodies)
        result['ids'][0].append(identifier)
        result['documents'][0].append(text)
        result['metadatas'][0].append(metadata)
        result['distances'][0].append(distances[identifier])
        if len(result['ids'][0]) >= limit:
            break
    return result
