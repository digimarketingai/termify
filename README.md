# 🔤 Termify

AI-powered bilingual terminology extractor for parallel texts. Extract Chinese-English term pairs using Mistral AI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)

## ✨ Features

- **Bilingual Extraction**: Extract Chinese-English terminology pairs from parallel texts
- **Smart Chunking**: Handles long documents by intelligently splitting and aligning text segments
- **Custom Commands**: Use natural language instructions to control extraction (e.g., "Extract only person names")
- **Multiple Export Formats**: CSV, JSON, TSV, and TBX (industry-standard terminology format)
- **Category Detection**: Automatically categorizes terms (medical, organization, place, technical, etc.)
- **Web Interface**: Easy-to-use Gradio UI with progress tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Mistral AI API key ([Get one free](https://console.mistral.ai/api-keys/))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/termify.git
cd termify

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Requirements

Create a `requirements.txt` file:

```
gradio>=4.0.0
openai>=1.0.0
```

## 📖 Usage

### Web Interface

1. Run `python app.py`
2. Open the provided URL in your browser
3. Paste your source text (Chinese)
4. Optionally paste target text (English translation) for better accuracy
5. Enter your Mistral API key
6. Click **Extract**

### Extraction Modes

#### Standard Mode (Keywords)

Enter simple keywords to focus extraction:

| Keyword | Focus Area |
|---------|------------|
| `social media` | Social platforms, accounts, websites |
| `medical` | Diseases, symptoms, procedures |
| `organization` | Government bodies, agencies |
| `place` | Locations, districts, countries |
| `technical` | Equipment, devices, procedures |
| `chemical` | Compounds, pesticides, ingredients |
| `date` | Dates, times, periods |

#### Custom Command Mode

Enter full instructions for precise control:

**English:**
- `Extract only person names and titles`
- `Find all organization names`
- `Get only dates and time expressions`
- `List all social media accounts mentioned`

**中文：**
- `只提取人名和職稱`
- `找出所有機構名稱`
- `只要日期和時間相關的詞彙`
- `列出所有提到的社群媒體帳號`

## 📤 Export Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| **CSV** | Comma-separated values | Excel, spreadsheets |
| **JSON** | Structured data format | APIs, programming |
| **TSV** | Tab-separated values | CAT tools, simple import |
| **TBX** | TermBase eXchange | Professional translation tools |

## 🔧 Configuration

### Environment Variables (Optional)

```bash
# Set default API key (not recommended for security)
export MISTRAL_API_KEY="your-api-key"
```

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Max Terms | 150 | 20-300 | Maximum terms to extract |
| Chunk Size | 1500 | - | Characters per segment |
| Max Chars | 20000 | - | Maximum input length |


## 🤖 How It Works

1. **Text Chunking**: Long texts are split into manageable segments using paragraph boundaries
2. **Alignment**: Source and target chunks are aligned proportionally
3. **Extraction**: Mistral AI analyzes each segment pair to identify terminology
4. **Validation**: Results are cleaned to remove duplicates and invalid entries
5. **Categorization**: Terms are automatically categorized by type
6. **Export**: Final glossary is formatted for your preferred output

## 📊 Example Output

| # | Source | Target | Category |
|:-:|:-------|:-------|:--------:|
| 1 | 登革熱 | Dengue fever | medical |
| 2 | 衛生署 | Department of Health | organization |
| 3 | 香港 | Hong Kong | place |
| 4 | 滅蚊 | Mosquito control | technical |

## ⚠️ Limitations

- Optimized for Chinese-English language pairs
- Requires active internet connection
- API rate limits may apply based on your Mistral plan
- Maximum input size: 20,000 characters per text field
