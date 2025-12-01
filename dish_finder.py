# Restaurant Dish Finder - Local RAG System
# No LLM API required! Uses local embeddings and search.

"""
This RAG system finds dishes from the Frankfurt Asian restaurants guide.
It uses:
- Sentence Transformers for embeddings (runs locally, free)
- FAISS for vector search (runs locally, free)
- Fuzzy matching for typo tolerance

No API key required!
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# For embeddings and search
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    ADVANCED_SEARCH = True
except ImportError:
    ADVANCED_SEARCH = False
    print("Note: Install sentence-transformers and faiss-cpu for semantic search")
    print("pip install sentence-transformers faiss-cpu")
    print("Falling back to keyword search...")

# For fuzzy matching
try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Note: Install rapidfuzz for better fuzzy matching")
    print("pip install rapidfuzz")


@dataclass
class MenuItem:
    """Represents a menu item"""
    dish_name: str
    price: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    restaurant: str = ""
    cuisine: str = ""
    price_range: str = ""
    address: str = ""


@dataclass 
class Restaurant:
    """Represents a restaurant"""
    name: str
    cuisine: str
    price_range: str
    address: str
    phone: str = ""
    website: str = ""
    description: str = ""
    menu_items: List[MenuItem] = field(default_factory=list)


class RestaurantKnowledgeBase:
    """
    Knowledge base for restaurant dishes.
    Parses the markdown document and creates searchable index.
    """
    
    def __init__(self):
        self.restaurants: List[Restaurant] = []
        self.menu_items: List[MenuItem] = []
        self.dish_index: Dict[str, List[MenuItem]] = {}  # dish_name -> items
        
        # For semantic search
        self.embeddings = None
        self.faiss_index = None
        self.model = None
        
    def load_from_markdown(self, markdown_path: str) -> None:
        """Parse the markdown document and extract restaurant/dish data"""
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse restaurants from markdown
        self._parse_markdown(content)
        
        # Build search index
        self._build_keyword_index()
        
        if ADVANCED_SEARCH:
            self._build_semantic_index()
            
        print(f"✅ Loaded {len(self.restaurants)} restaurants")
        print(f"✅ Indexed {len(self.menu_items)} menu items")
    
    def _parse_markdown(self, content: str) -> None:
        """Parse markdown content to extract restaurants and dishes"""
        
        # Split by restaurant sections (## numbered headers)
        sections = re.split(r'\n## \d+\.', content)
        
        for section in sections[1:]:  # Skip intro
            restaurant = self._parse_restaurant_section(section)
            if restaurant:
                self.restaurants.append(restaurant)
                self.menu_items.extend(restaurant.menu_items)
    
    def _parse_restaurant_section(self, section: str) -> Optional[Restaurant]:
        """Parse a single restaurant section"""
        
        lines = section.strip().split('\n')
        if not lines:
            return None
        
        # Extract restaurant name from first line
        name_match = re.match(r'^([^⭐]+)', lines[0])
        name = name_match.group(1).strip() if name_match else lines[0].strip()
        
        # Extract basic info
        cuisine = self._extract_field(section, r'\*\*Cuisine:\*\*\s*(.+)')
        price_range = self._extract_field(section, r'\*\*Price Range:\*\*\s*(.+)')
        address = self._extract_field(section, r'\*\*Address:\*\*\s*(.+)')
        phone = self._extract_field(section, r'\*\*Phone:\*\*\s*(.+)')
        website = self._extract_field(section, r'\*\*Website:\*\*\s*(.+)')
        
        restaurant = Restaurant(
            name=name,
            cuisine=cuisine or "Asian",
            price_range=price_range or "€€",
            address=address or "",
            phone=phone or "",
            website=website or ""
        )
        
        # Extract menu items
        menu_items = self._extract_menu_items(section, restaurant)
        restaurant.menu_items = menu_items
        
        return restaurant
    
    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        """Extract a field using regex"""
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None
    
    def _extract_menu_items(self, section: str, restaurant: Restaurant) -> List[MenuItem]:
        """Extract menu items from a section"""
        items = []
        
        # Pattern for menu items: "- Item Name - Price" or "- Item Name (description) - Price"
        # Also matches: "Item - €X" format
        patterns = [
            r'-\s+([^-€]+?)\s*[-–]\s*(€?\d+(?:[.,]\d+)?€?)',  # - Item - €X
            r'-\s+([^€\n]+?)\s+(€\d+(?:[.,]\d+)?)',  # - Item €X
            r'\*\*([^*]+)\*\*.*?(\d+(?:[.,]\d+)?€)',  # **Item** ... X€
        ]
        
        current_category = None
        
        for line in section.split('\n'):
            # Check for category headers
            cat_match = re.match(r'\*\*([^*:]+)(?:\s*\([^)]+\))?:\*\*', line)
            if cat_match:
                current_category = cat_match.group(1).strip()
                continue
            
            # Try to extract menu item
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    dish_name = match.group(1).strip()
                    price = match.group(2).strip() if len(match.groups()) > 1 else None
                    
                    # Clean up dish name
                    dish_name = re.sub(r'\s+', ' ', dish_name)
                    dish_name = dish_name.strip('- ')
                    
                    if len(dish_name) > 2:  # Skip very short matches
                        item = MenuItem(
                            dish_name=dish_name,
                            price=price,
                            category=current_category,
                            restaurant=restaurant.name,
                            cuisine=restaurant.cuisine,
                            price_range=restaurant.price_range,
                            address=restaurant.address
                        )
                        items.append(item)
                    break
        
        return items
    
    def _build_keyword_index(self) -> None:
        """Build keyword-based search index"""
        for item in self.menu_items:
            # Index by normalized dish name
            key = item.dish_name.lower()
            if key not in self.dish_index:
                self.dish_index[key] = []
            self.dish_index[key].append(item)
            
            # Also index individual words
            words = re.findall(r'\b\w+\b', key)
            for word in words:
                if len(word) > 2:
                    if word not in self.dish_index:
                        self.dish_index[word] = []
                    if item not in self.dish_index[word]:
                        self.dish_index[word].append(item)
    
    def _build_semantic_index(self) -> None:
        """Build semantic search index using sentence transformers"""
        if not ADVANCED_SEARCH:
            return
            
        print("Building semantic search index...")
        
        # Load model (small, fast model)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for all dishes
        dish_texts = [
            f"{item.dish_name} {item.category or ''} {item.cuisine}"
            for item in self.menu_items
        ]
        
        self.embeddings = self.model.encode(dish_texts, convert_to_numpy=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print("✅ Semantic index built")


class DishFinder:
    """
    RAG system for finding dishes in restaurants.
    No LLM API required!
    """
    
    def __init__(self, knowledge_base: RestaurantKnowledgeBase):
        self.kb = knowledge_base
    
    def find_dish(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Find dishes matching the query.
        
        Args:
            query: Dish name or description to search for
            top_k: Number of results to return
            
        Returns:
            List of matching dishes with restaurant info
        """
        results = []
        query_lower = query.lower().strip()
        
        # 1. Exact match
        exact_matches = self._exact_search(query_lower)
        results.extend(exact_matches)
        
        # 2. Fuzzy match
        if FUZZY_AVAILABLE and len(results) < top_k:
            fuzzy_matches = self._fuzzy_search(query_lower, top_k - len(results))
            for match in fuzzy_matches:
                if match not in results:
                    results.append(match)
        
        # 3. Semantic search
        if ADVANCED_SEARCH and len(results) < top_k:
            semantic_matches = self._semantic_search(query, top_k - len(results))
            for match in semantic_matches:
                if match not in results:
                    results.append(match)
        
        # 4. Keyword search fallback
        if len(results) < top_k:
            keyword_matches = self._keyword_search(query_lower, top_k - len(results))
            for match in keyword_matches:
                if match not in results:
                    results.append(match)
        
        return results[:top_k]
    
    def _exact_search(self, query: str) -> List[Dict]:
        """Exact match search"""
        results = []
        
        if query in self.kb.dish_index:
            for item in self.kb.dish_index[query]:
                results.append(self._item_to_result(item, score=1.0, match_type="exact"))
        
        return results
    
    def _fuzzy_search(self, query: str, limit: int) -> List[Dict]:
        """Fuzzy string matching"""
        results = []
        
        dish_names = list(set(item.dish_name.lower() for item in self.kb.menu_items))
        
        matches = process.extract(query, dish_names, scorer=fuzz.WRatio, limit=limit * 2)
        
        for match_name, score, _ in matches:
            if score > 60:  # Threshold
                for item in self.kb.menu_items:
                    if item.dish_name.lower() == match_name:
                        results.append(self._item_to_result(item, score=score/100, match_type="fuzzy"))
                        break
        
        return results[:limit]
    
    def _semantic_search(self, query: str, limit: int) -> List[Dict]:
        """Semantic similarity search"""
        results = []
        
        if not self.kb.model or not self.kb.faiss_index:
            return results
        
        # Encode query
        query_embedding = self.kb.model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.kb.faiss_index.search(
            query_embedding.astype('float32'), 
            limit
        )
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.kb.menu_items):
                item = self.kb.menu_items[idx]
                # Convert distance to similarity score
                score = 1 / (1 + distance)
                results.append(self._item_to_result(item, score=score, match_type="semantic"))
        
        return results
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """Simple keyword search"""
        results = []
        query_words = set(re.findall(r'\b\w+\b', query))
        
        scored_items = []
        for item in self.kb.menu_items:
            item_words = set(re.findall(r'\b\w+\b', item.dish_name.lower()))
            overlap = len(query_words & item_words)
            if overlap > 0:
                score = overlap / max(len(query_words), len(item_words))
                scored_items.append((item, score))
        
        # Sort by score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        for item, score in scored_items[:limit]:
            results.append(self._item_to_result(item, score=score, match_type="keyword"))
        
        return results
    
    def _item_to_result(self, item: MenuItem, score: float, match_type: str) -> Dict:
        """Convert MenuItem to result dict"""
        return {
            "dish_name": item.dish_name,
            "price": item.price,
            "category": item.category,
            "restaurant": item.restaurant,
            "cuisine": item.cuisine,
            "price_range": item.price_range,
            "address": item.address,
            "match_score": round(score, 3),
            "match_type": match_type
        }
    
    def format_results(self, results: List[Dict]) -> str:
        """Format results as readable text"""
        if not results:
            return "❌ No dishes found matching your query."
        
        output = []
        output.append(f"🍜 Found {len(results)} matching dish(es):\n")
        
        for i, result in enumerate(results, 1):
            output.append(f"{'='*50}")
            output.append(f"#{i} {result['dish_name']}")
            if result['price']:
                output.append(f"   💰 Price: {result['price']}")
            output.append(f"   🏠 Restaurant: {result['restaurant']}")
            output.append(f"   🍴 Cuisine: {result['cuisine']}")
            if result['category']:
                output.append(f"   📂 Category: {result['category']}")
            if result['address']:
                output.append(f"   📍 Address: {result['address']}")
            output.append(f"   🎯 Match: {result['match_type']} ({result['match_score']:.0%})")
            output.append("")
        
        return "\n".join(output)


def create_sample_data():
    """Create sample data if markdown file doesn't exist"""
    sample_md = """# Asian Restaurants Frankfurt

## 1. Góc Phố - Vietnamese Street Food ⭐⭐⭐⭐

**Cuisine:** Vietnamese
**Price Range:** € (Budget-friendly)
**Address:** Schärfengäßchen 6, 60311 Frankfurt am Main

### Menu

**Nudelsuppen (Pho & Soups):**
- Pho Bo (Beef noodle soup) - 14€
- Pho Ga (Chicken noodle soup) - 13€
- Bun Bo Hue - 15€

**Vorspeisen (Starters):**
- Cha Gio (Spring rolls) - 6€
- Goi Cuon (Summer rolls) - 6€

**Reisgerichte (Rice):**
- Com Ga Nuong (Grilled chicken rice) - 14€

---

## 2. Thong Thai ⭐⭐⭐⭐

**Cuisine:** Thai
**Price Range:** € (Budget-friendly)
**Address:** Meisengasse 12, 60313 Frankfurt

### Menu

**Main Dishes:**
- Pad Thai - 7€
- Green Curry (Kiow-Wan-Gai) - 7€
- Red Curry (Gaeng-Daeng-Gai) - 7€
- Massaman Curry - 8€

**Soups:**
- Tom Yam Gai - 3€
- Tom Kha - 3€

---

## 3. Zenzakan ⭐⭐⭐⭐

**Cuisine:** Pan-Asian
**Price Range:** €€€
**Address:** Taunusanlage 15, 60325 Frankfurt

### Menu

**Sushi & Sashimi:**
- Salmon Sashimi - 18€
- Tuna Sashimi - 22€
- Dragon Roll - 16€

**Grill:**
- Wagyu Beef - 45€
- Black Pepper Beef - 28€

---

## 4. Pak Choi ⭐⭐⭐⭐

**Cuisine:** Chinese (Szechuan)
**Price Range:** €
**Address:** Dreieichstraße 7, 60594 Frankfurt

### Menu

**Specialties:**
- Kung Pao Chicken - 12€
- Mapo Tofu - 10€
- Dan Dan Noodles - 9€
- Szechuan Dumplings - 8€
"""
    return sample_md


# Interactive CLI
def main():
    """Main function - Interactive dish finder"""
    
    print("="*60)
    print("🍜 Frankfurt Asian Restaurant Dish Finder")
    print("   Local RAG System - No API Key Required!")
    print("="*60)
    print()
    
    # Check for markdown file
    md_path = Path("asian-restaurants-frankfurt-guide.md")
    
    if not md_path.exists():
        # Try alternate locations
        alternate_paths = [
            Path("/mnt/user-data/outputs/asian-restaurants-frankfurt-guide.md"),
            Path("restaurants.md"),
        ]
        for alt_path in alternate_paths:
            if alt_path.exists():
                md_path = alt_path
                break
    
    if not md_path.exists():
        print("📝 Creating sample restaurant data...")
        sample_data = create_sample_data()
        md_path = Path("restaurants_sample.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        print(f"   Created: {md_path}")
    
    # Initialize knowledge base
    print(f"\n📚 Loading knowledge base from: {md_path}")
    kb = RestaurantKnowledgeBase()
    kb.load_from_markdown(str(md_path))
    
    # Initialize finder
    finder = DishFinder(kb)
    
    # Interactive loop
    print("\n" + "="*60)
    print("Ready to search! Type a dish name or 'quit' to exit.")
    print("Examples: 'pho', 'pad thai', 'sushi', 'curry', 'dumplings'")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("🔍 Search dish: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Search
            results = finder.find_dish(query, top_k=5)
            
            # Display results
            print()
            print(finder.format_results(results))
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
