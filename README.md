# 🍜 Restaurant Dish Finder - Local RAG System

A simple RAG (Retrieval-Augmented Generation) system for finding dishes in Frankfurt Asian restaurants.

## ✨ Key Features

- **No LLM API required!** - Runs completely locally
- **No API keys needed** - Free to use
- **Two versions available:**
  - `simple_dish_finder.py` - Zero dependencies (pure Python)
  - `dish_finder.py` - Enhanced with semantic search (optional ML libraries)

## 🚀 Quick Start

### Option 1: Simple Version (No Dependencies)

```bash
# Just run it!
python simple_dish_finder.py
```

### Option 2: Enhanced Version (With Semantic Search)

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python dish_finder.py
```

## 📁 Files

| File | Description |
|------|-------------|
| `simple_dish_finder.py` | Zero-dependency version (pure Python) |
| `dish_finder.py` | Enhanced version with semantic search |
| `requirements.txt` | Dependencies for enhanced version |
| `asian-restaurants-frankfurt-guide.md` | Restaurant data (optional) |

## 🔍 How It Works

### Search Methods

1. **Exact Match** - Direct dish name match
2. **Partial Match** - Substring matching
3. **Fuzzy Match** - Handles typos (using difflib or rapidfuzz)
4. **Keyword Match** - Word-based matching
5. **Semantic Search** - Meaning-based search (enhanced version only)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Query                       │
│                    "pad thai"                       │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              Search Pipeline                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐   │
│  │ Exact   │→│ Partial │→│ Fuzzy   │→│ Keyword  │   │
│  │ Match   │ │ Match   │ │ Match   │ │ Match    │   │
│  └─────────┘ └─────────┘ └─────────┘ └──────────┘   │
│                      │                              │
│                      ▼ (if ML libraries installed)  │
│              ┌──────────────┐                       │
│              │   Semantic   │                       │
│              │   Search     │                       │
│              └──────────────┘                       │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              Ranked Results                         │
│  #1 Phad Thai @ Thong Thai - 7€ (exact, 100%)       │
│  #2 Pad Thai @ Zenzakan - 18€ (partial, 90%)        │
└─────────────────────────────────────────────────────┘
```

## 💻 Usage Examples

### Interactive Mode

```
🔍 Search: pho

🍜 Found 3 result(s) for 'pho':
=======================================================

#1 Pho Bo (Beef noodle soup)
   💰 Price: 14€
   🏠 Restaurant: Góc Phố - Vietnamese Street Food
   🍴 Cuisine: Vietnamese
   📂 Category: Nudelsuppen (Pho & Soups)
   📍 Address: Schärfengäßchen 6, 60311 Frankfurt
   🎯 Match: exact (100%)

#2 Pho Ga (Chicken noodle soup)
   💰 Price: 13€
   🏠 Restaurant: Góc Phố - Vietnamese Street Food
   🎯 Match: partial (90%)
```

### Programmatic Usage

```python
from simple_dish_finder import SimpleRAG

# Initialize
rag = SimpleRAG()
rag.load_from_markdown("restaurants.md")

# Search
results = rag.search("curry", top_k=5)

for item, score, match_type in results:
    print(f"{item.dish_name} @ {item.restaurant} - {item.price}")
```

## 🤔 Why No LLM API?

For this specific use case, an LLM is overkill because:

| Task | LLM Needed? | Our Solution |
|------|-------------|--------------|
| Find dish in document | ❌ | Keyword/semantic search |
| Handle typos | ❌ | Fuzzy string matching |
| Return restaurant info | ❌ | Structured data lookup |
| Natural language understanding | ❌ | Simple keyword extraction |

**When you WOULD need an LLM:**
- Answering complex questions about the food
- Generating recommendations based on preferences
- Having a conversation about restaurants
- Summarizing reviews

## 📊 Search Quality Comparison

| Method | Speed | Typo Tolerance | Semantic Understanding |
|--------|-------|----------------|------------------------|
| Exact Match | ⚡⚡⚡ | ❌ | ❌ |
| Fuzzy Match | ⚡⚡ | ✅ | ❌ |
| Keyword Match | ⚡⚡⚡ | ⚠️ | ❌ |
| Semantic Search | ⚡ | ✅ | ✅ |

## 🛠️ Customization

### Add Your Own Restaurant Data

Create a markdown file with this structure:

```markdown
## 1. Restaurant Name ⭐⭐⭐⭐

**Cuisine:** Thai
**Price Range:** €€
**Address:** Street 123, City

### Menu

**Category:**
- Dish Name - Price€
- Another Dish - Price€
```

### Adjust Search Sensitivity

In `simple_dish_finder.py`:

```python
# Fuzzy match threshold (0.0 - 1.0)
if ratio > 0.5:  # Lower = more results, higher = stricter

# Keyword match threshold
if score > 0.3:  # Lower = more results
```

## 📝 License

MIT License - Feel free to use and modify!

## 🙏 Credits

Restaurant data compiled from various sources including:
- TripAdvisor
- Yelp
- Restaurant websites
- Wolt/Lieferando

---

**No API keys. No costs. Just search!** 🍜
