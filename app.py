"""
Termify - AI-powered bilingual terminology extractor
https://github.com/digimarketingai/termify
"""

import gradio as gr
import openai
import json
import re
import time


def get_client(token=""):
    """Initialize Mistral API client."""
    return openai.OpenAI(
        base_url="https://api.mistral.ai/v1",
        api_key=token if token.strip() else "unused",
    )


# Configuration constants
MAX_CHARS = 20000
CHUNK_SIZE = 2000  # Increased chunk size for better context


def smart_chunk(text, size=2000):
    """Split text into chunks using paragraph boundaries."""
    if not text or len(text) <= size:
        return [text] if text else []
    
    chunks = []
    paragraphs = re.split(r'\n\s*\n', text)
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n" if len(para) <= size else para[:size]
    
    if current:
        chunks.append(current.strip())
    
    return chunks if chunks else [text[:size]]


def align_chunks(source_chunks, target_chunks):
    """Align source and target chunks proportionally."""
    if not target_chunks:
        return [(s, "") for s in source_chunks]
    
    if len(source_chunks) == len(target_chunks):
        return list(zip(source_chunks, target_chunks))
    
    full_target = "\n\n".join(target_chunks)
    target_len = len(full_target)
    source_lengths = [len(s) for s in source_chunks]
    total_source = sum(source_lengths)
    
    aligned = []
    pos = 0
    for i, src in enumerate(source_chunks):
        ratio = source_lengths[i] / total_source
        chunk_len = int(ratio * target_len)
        end_pos = min(pos + chunk_len, target_len)
        
        if end_pos < target_len:
            boundary = full_target.rfind('\n', pos, end_pos + 200)
            if boundary > pos:
                end_pos = boundary
        
        aligned.append((src, full_target[pos:end_pos].strip()))
        pos = end_pos
    
    return aligned


def parse_terms(content):
    """Parse JSON term data from API response."""
    terms = []
    
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```\w*\n?', '', content)
        content = re.sub(r'\n?```$', '', content)
    
    try:
        match = re.search(r'\[[\s\S]*\]', content)
        if match:
            data = json.loads(match.group())
            for item in data:
                if isinstance(item, dict) and item.get('source'):
                    src = str(item.get('source', '')).strip()
                    tgt = str(item.get('target', item.get('translation', ''))).strip()
                    cat = str(item.get('category', 'general')).strip().lower()
                    
                    # Skip null-like values
                    if not tgt or tgt.lower() in ['null', 'none', 'n/a', 'undefined']:
                        continue
                    
                    if src == tgt and re.match(r'^[A-Za-z\s]+$', src):
                        continue
                    if any(x in src.lower() for x in ['extract', 'priority', 'category', 'include', 'skip', 'rules']):
                        continue
                    
                    if src and len(src) >= 2:
                        terms.append({'source': src, 'target': tgt, 'category': cat})
            return terms
    except:
        pass
    
    for obj_str in re.findall(r'\{[^{}]+\}', content):
        try:
            obj = json.loads(obj_str)
            if obj.get('source'):
                src = str(obj.get('source', '')).strip()
                tgt = str(obj.get('target', '')).strip()
                
                # Skip null-like values
                if not tgt or tgt.lower() in ['null', 'none', 'n/a', 'undefined']:
                    continue
                
                if src == tgt and re.match(r'^[A-Za-z\s]+$', src):
                    continue
                    
                terms.append({
                    'source': src,
                    'target': tgt,
                    'category': str(obj.get('category', 'general')).strip().lower()
                })
        except:
            pass
    
    return terms


def is_custom_command(focus_text):
    """
    Detect if the focus field contains a custom command/prompt.
    Returns True if user wants to use custom extraction logic.
    """
    if not focus_text or not focus_text.strip():
        return False
    
    focus_lower = focus_text.lower().strip()
    
    # Command indicators - words that suggest a custom instruction
    command_indicators = [
        # English command words
        'extract', 'find', 'get', 'list', 'identify', 'locate', 'search',
        'only', 'just', 'specifically', 'exclusively',
        'please', 'i want', 'i need', 'give me', 'show me',
        'focus on', 'look for', 'pull out', 'pick out',
        'include', 'exclude', 'ignore', 'skip',
        'all', 'every', 'any', 'must', 'should',
        # Chinese command words
        '提取', '找', '找出', '列出', '識別', '搜尋', '搜索',
        '只要', '僅', '專門', '特別',
        '請', '我要', '我需要', '給我', '顯示',
        '專注', '尋找', '挑出',
        '包含', '排除', '忽略', '跳過',
        '所有', '每個', '任何', '必須', '應該',
        # Pattern indicators
        'term', 'terms', 'word', 'words', 'phrase', 'phrases',
        'name', 'names', 'entity', 'entities',
        '術語', '詞', '詞彙', '名稱', '實體',
    ]
    
    # Check for command indicators
    for indicator in command_indicators:
        if indicator in focus_lower:
            return True
    
    # Check for sentence-like structure
    if len(focus_text.strip()) > 20 and ' ' in focus_text:
        return True
    
    # Check for punctuation that suggests a sentence/command
    if any(p in focus_text for p in ['。', '，', '.', ',', '!', '！', '?', '？']):
        return True
    
    return False


def get_focus_instruction(focus):
    """Get predefined focus instruction for simple keywords."""
    if not focus or not focus.strip():
        return ""
    
    focus_lower = focus.lower().strip()
    
    focus_map = {
        "social media": "Pay special attention to social media platforms, Facebook pages, Instagram accounts, YouTube channels, websites.",
        "medical": "Pay special attention to diseases, symptoms, medical procedures, health terms.",
        "organization": "Pay special attention to government departments, agencies, official bodies.",
        "place": "Pay special attention to locations, districts, trails, parks, countries.",
        "technical": "Pay special attention to equipment, devices, machinery, technical procedures.",
        "chemical": "Pay special attention to chemical compounds, pesticides, larvicides, active ingredients.",
        "date": "Pay special attention to dates, times, years, months, days, periods."
    }
    
    for key, instruction in focus_map.items():
        if key in focus_lower:
            return instruction
    
    return f"Pay special attention to terms related to: {focus}"


def extract_chunk_custom(source, target, custom_prompt, client):
    """
    Extract terms using custom user prompt - follows user instructions directly.
    Improved to better match translations from parallel target text.
    """
    if target:
        max_target = min(len(target), 4000 - len(source))
        target_truncated = target[:max_target]
        
        prompt = f"""You are a bilingual terminology extractor working with PARALLEL Chinese-English texts (they are translations of each other).

<source_chinese>
{source}
</source_chinese>

<target_english>
{target_truncated}
</target_english>

USER INSTRUCTION: {custom_prompt}

CRITICAL MATCHING RULES:
1. Follow the user's instruction to identify WHICH terms to extract from the Chinese text
2. For EVERY Chinese term, you MUST find its EXACT English translation from the target_english text above
3. The texts are parallel translations - every Chinese term HAS a corresponding English term in the English text
4. Search carefully in the English text - the translation IS there
5. NEVER use "null", "N/A", or leave target empty

EXAMPLES of correct matching:
- 地政總署 → "Lands Department" (find it in English text)
- 渠務署 → "Drainage Services Department" (find it in English text)
- 葵青民政事務處 → "Kwai Tsing District Office" (find it in English text)
- 衞生防護中心 → "Centre for Health Protection" (find it in English text)

Output ONLY a JSON array with terms you found AND their English translations from the text:
[{{"source":"中文術語","target":"English from target text","category":"category"}}]"""

    else:
        prompt = f"""You are a bilingual terminology extractor.

<chinese_text>
{source}
</chinese_text>

USER INSTRUCTION: {custom_prompt}

Based on the user's instruction, extract the requested terms and provide accurate English translations.
NEVER use "null" - always provide a real English translation.

Output ONLY a JSON array:
[{{"source":"中文術語","target":"English translation","category":"type"}}]"""

    try:
        resp = client.chat.completions.create(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "You are a precise bilingual terminology extractor. When given parallel texts, you MUST match Chinese terms with their English translations from the English text. The English translation is ALWAYS present in the parallel text - search carefully. NEVER output null or empty translations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2500,
        )
        
        content = resp.choices[0].message.content.strip()
        terms = parse_terms(content)
        return terms, content
        
    except Exception as e:
        return [], str(e)


def extract_chunk(source, target, focus, client):
    """Standard extraction with predefined logic."""
    focus_instruction = get_focus_instruction(focus)
    
    term_target = "40-60"
    
    if target:
        max_target = min(len(target), 4000 - len(source))
        target_truncated = target[:max_target]
        
        prompt = f"""You are a bilingual terminology extractor. Extract Chinese-English term pairs from these PARALLEL texts (they are translations of each other).

<source_chinese>
{source}
</source_chinese>

<target_english>
{target_truncated}
</target_english>

Instructions:
- Extract {term_target} terminology pairs
- Match Chinese terms with their English translations FROM THE ENGLISH TEXT ABOVE
- Include: proper nouns, technical terms, organizations, places, dates/times, chemicals, medical terms
- {focus_instruction if focus_instruction else "Extract all types of terminology"}
- NEVER use "null" - the English translation is in the target text
- Use categories: medical, organization, place, social, technical, chemical, date, general

Output ONLY a JSON array:
[{{"source":"中文術語","target":"English term from text","category":"type"}}]"""

    else:
        prompt = f"""You are a bilingual terminology extractor. Extract key Chinese terms with English translations.

<chinese_text>
{source}
</chinese_text>

Instructions:
- Extract {term_target} terms with accurate English translations
- Include: proper nouns, technical terms, organizations, places, dates/times, chemicals, medical terms
- {focus_instruction if focus_instruction else "Extract all types of terminology"}
- NEVER use "null" - always provide real translations
- Use categories: medical, organization, place, social, technical, chemical, date, general

Output ONLY a JSON array:
[{{"source":"中文術語","target":"English term","category":"type"}}]"""

    try:
        resp = client.chat.completions.create(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "You extract terminology from texts. Output only valid JSON arrays. Never include instruction text in output. Never use null for translations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2500,
        )
        
        content = resp.choices[0].message.content.strip()
        terms = parse_terms(content)
        return terms, content
        
    except Exception as e:
        return [], str(e)


def dedupe(terms):
    """Remove duplicate terms, keeping the best translation."""
    seen = {}
    for t in terms:
        key = t['source'].lower()
        if key not in seen:
            seen[key] = t
        elif len(t['target']) > len(seen[key]['target']):
            seen[key] = t
    return list(seen.values())


def validate_terms(terms):
    """Filter out invalid or garbage terms."""
    valid = []
    
    garbage_patterns = [
        r'^[A-Za-z\s]{2,}$',
        r'extract|priority|category|include|skip|rules|instructions',
        r'^[\d\.\s]+$',
    ]
    
    invalid_targets = ['null', 'none', 'n/a', 'undefined', 'nil', '']
    
    for t in terms:
        src = t['source'].strip()
        tgt = t['target'].strip() if t.get('target') else ''
        
        # Skip if source or target is empty/null
        if not src or not tgt:
            continue
        
        # Skip if target is a null-like value
        if tgt.lower() in invalid_targets:
            continue
        
        # Skip if source equals target and it's just English text
        if src.lower() == tgt.lower() and re.match(r'^[A-Za-z0-9\s\-]+$', src):
            if len(src) <= 6 or src.upper() == src:  # Allow acronyms
                pass
            else:
                continue
        
        # Skip garbage patterns in source
        if any(re.search(p, src.lower()) for p in garbage_patterns[1:2]):
            continue
        
        # Skip long English-only source
        if re.match(r'^[A-Za-z\s]{10,}$', src):
            continue
            
        valid.append(t)
    
    return valid


def extract_terms(source_text, target_text, focus, max_terms, api_token, progress=gr.Progress()):
    """Main extraction function."""
    if not source_text or not source_text.strip():
        return "❌ Please enter source text. | 請輸入來源文本。", "", gr.update(visible=False), ""
    
    if not api_token or not api_token.strip():
        return "❌ Mistral API key is required. | 需要 Mistral API 密鑰。", "", gr.update(visible=False), ""
    
    client = get_client(api_token)
    
    source_text = source_text.strip()[:MAX_CHARS]
    target_text = target_text.strip()[:MAX_CHARS] if target_text else ""
    focus = focus.strip() if focus else ""
    
    # Detect if using custom command mode
    use_custom_mode = is_custom_command(focus)
    
    progress(0.05, desc="📝 Preparing...")
    
    if use_custom_mode:
        progress(0.1, desc="🎯 Custom command detected! Following your instructions...")
    
    source_chunks = smart_chunk(source_text, CHUNK_SIZE)
    target_chunks = smart_chunk(target_text, CHUNK_SIZE) if target_text else []
    aligned_pairs = align_chunks(source_chunks, target_chunks)
    
    progress(0.1, desc=f"🔄 Processing {len(aligned_pairs)} segment(s)...")
    
    all_terms = []
    debug_logs = []
    start_time = time.time()
    
    mode_label = "CUSTOM COMMAND" if use_custom_mode else "STANDARD"
    debug_logs.append(f"Mode: {mode_label}\n")
    debug_logs.append(f"API: Mistral (mistral-small-latest)\n")
    if use_custom_mode:
        debug_logs.append(f"User Command: {focus}\n")
    
    for i, (src, tgt) in enumerate(aligned_pairs):
        progress(0.1 + 0.7 * ((i + 1) / len(aligned_pairs)), 
                desc=f"🤖 Segment {i+1}/{len(aligned_pairs)}...")
        
        # Use custom extraction if in custom mode
        if use_custom_mode:
            terms, raw = extract_chunk_custom(src, tgt, focus, client)
        else:
            terms, raw = extract_chunk(src, tgt, focus, client)
        
        debug_logs.append(f"""
=== Segment {i+1} ===
Source: {len(src)} chars | Target: {len(tgt)} chars
Raw terms: {len(terms)}
Response preview: {raw[:600]}...
""")
        
        all_terms.extend(terms)
        
        if i < len(aligned_pairs) - 1:
            time.sleep(0.5)
    
    progress(0.85, desc="🔍 Cleaning results...")
    
    valid_terms = validate_terms(all_terms)
    unique_terms = dedupe(valid_terms)
    raw_count = len(unique_terms)
    
    # Sort terms by category and source
    unique_terms.sort(key=lambda t: (t.get('category', 'zzz'), t['source']))
    
    final_terms = unique_terms[:max_terms]
    
    elapsed = time.time() - start_time
    
    debug_log = f"""=== EXTRACTION SUMMARY ===
Mode: {mode_label}
API: Mistral (mistral-small-latest)
Focus/Command: {focus if focus else 'None'}
Segments: {len(aligned_pairs)}
Time: {elapsed:.1f}s

Raw extracted: {len(all_terms)}
After validation: {len(valid_terms)}
After dedupe: {raw_count}
Final: {len(final_terms)}

{"".join(debug_logs)}
"""
    
    if not final_terms:
        msg = f"⚠️ No terms found"
        if use_custom_mode:
            msg += f" matching your command.\n💡 Try a different instruction or simpler request."
        return msg, "", gr.update(visible=False), debug_log
    
    progress(0.95, desc="📊 Formatting...")
    
    # Build result table
    table = "| # | Source | Target | Category |\n|:---:|:---|:---|:---:|\n"
    for i, t in enumerate(final_terms, 1):
        src = t['source'].replace('|', '∣')
        tgt = t['target'].replace('|', '∣')
        cat = t.get('category', 'general')
        table += f"| {i} | {src} | {tgt} | {cat} |\n"
    
    # Build CSV
    csv_lines = ["Source,Target,Category"]
    for t in final_terms:
        src_csv = t["source"].replace('"', '""')
        tgt_csv = t["target"].replace('"', '""')
        csv_lines.append(f'"{src_csv}","{tgt_csv}","{t.get("category", "general")}"')
    csv_content = "\n".join(csv_lines)
    
    progress(1.0, desc="✅ Done!")
    
    # Build result message
    mode_note = "🎯 **Custom Mode**" if use_custom_mode else ""
    
    result = f"✅ **{len(final_terms)} terms** extracted in {elapsed:.1f}s\n{mode_note}\n\n{table}"
    
    return result, csv_content, gr.update(visible=True), debug_log


def save_file(csv_content, fmt):
    """Export terms to various file formats."""
    if not csv_content:
        return None
    
    lines = csv_content.strip().split('\n')[1:]
    terms = []
    for line in lines:
        parts = line.split('","')
        if len(parts) >= 3:
            terms.append({
                'source': parts[0].strip('"'),
                'target': parts[1],
                'category': parts[2].strip('"')
            })
    
    paths = {
        "csv": "/tmp/termify_glossary.csv",
        "json": "/tmp/termify_glossary.json",
        "tsv": "/tmp/termify_glossary.tsv",
        "tbx": "/tmp/termify_glossary.tbx"
    }
    path = paths.get(fmt)
    
    if fmt == "csv":
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write(csv_content)
    elif fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"terms": terms, "count": len(terms)}, f, indent=2, ensure_ascii=False)
    elif fmt == "tsv":
        with open(path, "w", encoding="utf-8") as f:
            for t in terms:
                f.write(f"{t['source']}\t{t['target']}\n")
    elif fmt == "tbx":
        tbx = '<?xml version="1.0" encoding="UTF-8"?>\n'
        tbx += '<!DOCTYPE martif SYSTEM "TBXcoreStructV02.dtd">\n'
        tbx += '<martif type="TBX" xml:lang="en">\n'
        tbx += '  <martifHeader>\n'
        tbx += '    <fileDesc>\n'
        tbx += '      <titleStmt>\n'
        tbx += '        <title>Termify Glossary Export</title>\n'
        tbx += '      </titleStmt>\n'
        tbx += '    </fileDesc>\n'
        tbx += '  </martifHeader>\n'
        tbx += '  <text>\n'
        tbx += '    <body>\n'
        for i, t in enumerate(terms):
            src_escaped = t["source"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            tgt_escaped = t["target"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            tbx += f'      <termEntry id="t{i+1}">\n'
            tbx += f'        <descrip type="subjectField">{t.get("category", "general")}</descrip>\n'
            tbx += f'        <langSet xml:lang="zh">\n'
            tbx += f'          <tig>\n'
            tbx += f'            <term>{src_escaped}</term>\n'
            tbx += f'          </tig>\n'
            tbx += f'        </langSet>\n'
            tbx += f'        <langSet xml:lang="en">\n'
            tbx += f'          <tig>\n'
            tbx += f'            <term>{tgt_escaped}</term>\n'
            tbx += f'          </tig>\n'
            tbx += f'        </langSet>\n'
            tbx += f'      </termEntry>\n'
        tbx += '    </body>\n'
        tbx += '  </text>\n'
        tbx += '</martif>'
        with open(path, "w", encoding="utf-8") as f:
            f.write(tbx)
    
    return path


def clear_all():
    """Reset all form fields."""
    return "", "", "", 150, "", "📋 Ready | 準備就緒", "", gr.update(visible=False)


# ========== GRADIO UI ==========

with gr.Blocks(title="Termify - Terminology Extractor", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🔤 Termify")
    gr.Markdown("**AI-powered bilingual terminology extractor** | 智能雙語術語提取工具")
    gr.Markdown("*Powered by Mistral AI* | *By [digimarketingai](https://github.com/digimarketingai/termify)*")
    
    with gr.Row():
        with gr.Column():
            source_box = gr.Textbox(
                label="📄 Source Text (Required) | 來源文本（必填）", 
                lines=10, 
                placeholder="Paste Chinese source text here...\n貼上中文來源文本..."
            )
        with gr.Column():
            target_box = gr.Textbox(
                label="📝 Target Text (Optional) | 目標文本（選填）", 
                lines=10, 
                placeholder="Paste English translation for better accuracy...\n貼上英文翻譯以提高準確性..."
            )
    
    with gr.Row():
        focus_box = gr.Textbox(
            label="🎯 Focus / Custom Command | 提取重點 / 自訂指令", 
            placeholder="Keywords: social media, medical, place | OR | Custom: 'Extract only person names' / '只提取人名'",
            info="💡 Enter keywords OR full commands. Commands are followed directly. | 輸入關鍵字或完整指令，系統會直接遵循指令。",
            scale=2
        )
        max_slider = gr.Slider(
            label="Max Terms | 最大術語數", 
            minimum=20, 
            maximum=300, 
            value=150, 
            step=10, 
            scale=1
        )
    
    with gr.Accordion("🔑 Mistral API Key (Required) | Mistral API 密鑰（必填）", open=True):
        token_box = gr.Textbox(
            label="Mistral API Key", 
            placeholder="Enter your Mistral API key here...",
            type="password"
        )
        gr.Markdown("🔗 [Get your free Mistral API key →](https://console.mistral.ai/api-keys/)")
    
    with gr.Row():
        extract_btn = gr.Button("🚀 Extract Terms | 提取術語", variant="primary", scale=2)
        clear_btn = gr.Button("🗑️ Clear All | 清除全部", scale=1)
    
    result_box = gr.Markdown("📋 Ready | 準備就緒")
    csv_state = gr.State("")
    
    download_row = gr.Row(visible=False)
    with download_row:
        csv_btn = gr.Button("📥 CSV")
        json_btn = gr.Button("📥 JSON")
        tsv_btn = gr.Button("📥 TSV")
        tbx_btn = gr.Button("📥 TBX")
    
    file_output = gr.File(label="Download", visible=True)
    
    with gr.Accordion("🔧 Debug Log | 除錯日誌", open=False):
        debug_box = gr.Textbox(lines=15, show_copy_button=True)
    
    with gr.Accordion("💡 Tips & Examples | 使用提示與範例", open=False):
        gr.Markdown("""
## 🆕 Custom Command Mode | 自訂指令模式

Enter full commands in the Focus field for precise control:

### English Examples:
- `Extract only person names and titles`
- `Find all organization names`
- `Get only dates and time expressions`
- `Extract medical terms related to dengue fever`
- `List all social media accounts mentioned`

### 中文範例：
- `只提取人名和職稱`
- `找出所有機構名稱`
- `只要日期和時間相關的詞彙`
- `提取與登革熱相關的醫學術語`
- `列出所有提到的社群媒體帳號`

---

## Standard Mode | 標準模式

**Simple keywords** trigger standard extraction with focus:
- `social media` → Prioritizes social platforms
- `medical` → Prioritizes medical terms
- `organization` → Prioritizes organizations
- `place` → Prioritizes locations
- `chemical` → Prioritizes chemical terms
- `date` → Prioritizes dates/times

---

## Tips | 提示

- **With target text**: Provides more accurate translations
- **Max Terms**: Limits results to top N terms
- **Export formats**: CSV (Excel), JSON (APIs), TSV (CAT tools), TBX (professional)
        """)
    
    # Event handlers
    extract_btn.click(
        extract_terms, 
        inputs=[source_box, target_box, focus_box, max_slider, token_box],
        outputs=[result_box, csv_state, download_row, debug_box]
    )
    
    clear_btn.click(
        clear_all, 
        outputs=[source_box, target_box, focus_box, max_slider, token_box, result_box, csv_state, download_row]
    )
    
    csv_btn.click(lambda c: save_file(c, "csv"), inputs=[csv_state], outputs=[file_output])
    json_btn.click(lambda c: save_file(c, "json"), inputs=[csv_state], outputs=[file_output])
    tsv_btn.click(lambda c: save_file(c, "tsv"), inputs=[csv_state], outputs=[file_output])
    tbx_btn.click(lambda c: save_file(c, "tbx"), inputs=[csv_state], outputs=[file_output])


# ========== LAUNCH ==========

if __name__ == "__main__":
    demo.launch(share=True)
