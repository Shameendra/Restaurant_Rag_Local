#!/usr/bin/env python3
"""
Restaurant Dish Finder - Simple Version
======================================
No external dependencies required! Pure Python implementation.

This is a lightweight RAG system that finds dishes from the 
Frankfurt Asian restaurants guide using:
- Keyword matching
- Simple fuzzy matching (built-in)
- No API keys needed
- No ML libraries needed
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher


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
    menu_items: List[MenuItem] = field(default_factory=list)


class SimpleRAG:
    """
    Simple RAG system for restaurant dish lookup.
    No external dependencies - pure Python!
    """
    
    def __init__(self):
        self.restaurants: List[Restaurant] = []
        self.menu_items: List[MenuItem] = []
        self.dish_index: Dict[str, List[MenuItem]] = {}
    
    def load_from_markdown(self, filepath: str) -> None:
        """Load and parse the markdown document"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self._parse_markdown(content)
        self._build_index()
        
        print(f"✅ Loaded {len(self.restaurants)} restaurants")
        print(f"✅ Indexed {len(self.menu_items)} menu items")
    
    def load_from_string(self, content: str) -> None:
        """Load from string content"""
        self._parse_markdown(content)
        self._build_index()
    
    def _parse_markdown(self, content: str) -> None:
        """Parse markdown to extract restaurants and dishes"""
        
        # Split by restaurant sections
        sections = re.split(r'\n## \d+\.', content)
        
        for section in sections[1:]:
            restaurant = self._parse_restaurant(section)
            if restaurant and restaurant.menu_items:
                self.restaurants.append(restaurant)
                self.menu_items.extend(restaurant.menu_items)
    
    def _parse_restaurant(self, section: str) -> Optional[Restaurant]:
        """Parse a restaurant section"""
        lines = section.strip().split('\n')
        if not lines:
            return None
        
        # Get restaurant name
        name = re.sub(r'[⭐\*\(\)0-9./]+', '', lines[0]).strip()
        
        # Extract info using regex
        def extract(pattern):
            match = re.search(pattern, section, re.IGNORECASE)
            return match.group(1).strip() if match else ""
        
        restaurant = Restaurant(
            name=name,
            cuisine=extract(r'\*\*Cuisine:\*\*\s*(.+)'),
            price_range=extract(r'\*\*Price Range:\*\*\s*(.+)'),
            address=extract(r'\*\*Address:\*\*\s*(.+)'),
            phone=extract(r'\*\*Phone:\*\*\s*(.+)'),
            website=extract(r'\*\*Website:\*\*\s*(.+)')
        )
        
        # Extract menu items
        restaurant.menu_items = self._extract_dishes(section, restaurant)
        
        return restaurant
    
    def _extract_dishes(self, section: str, restaurant: Restaurant) -> List[MenuItem]:
        """Extract dishes from section"""
        items = []
        current_category = None
        
        for line in section.split('\n'):
            # Check for category
            cat_match = re.match(r'\*\*([^*:]+).*:\*\*', line)
            if cat_match:
                current_category = cat_match.group(1).strip()
                continue
            
            # Match dish patterns - expanded
            patterns = [
                # Standard format: - Dish Name - 14€
                r'-\s+([^-€]+?)\s*[-–]\s*(€?\d+(?:[.,]\d+)?€?)',
                # Format: - Dish Name 14€
                r'-\s+([^€\n]+?)\s+(€\d+(?:[.,]\d+)?)',
                # Format: - Dish - €X.XX
                r'-\s+([A-Za-z][^-\n]{2,}?)\s+-\s+(\d+€)',
                # Format with parentheses: - Dish (description) - 14€
                r'-\s+([^(]+(?:\([^)]+\))?)\s*[-–]\s*(€?\d+(?:[.,]\d+)?)',
                # Simple: Dish Name - €XX
                r'([A-Za-z][A-Za-z\s]+)\s*[-–]\s*(€?\d+(?:[.,]\d+)?€?)',
                # Bullet without dash: • Dish - Price
                r'•\s+([^-€]+?)\s*[-–]\s*(€?\d+(?:[.,]\d+)?€?)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    dish_name = match.group(1).strip(' -*()')
                    price = match.group(2).strip() if len(match.groups()) > 1 else None
                    
                    # Clean up
                    dish_name = re.sub(r'\s+', ' ', dish_name)
                    dish_name = dish_name.strip()
                    
                    # Skip invalid entries
                    if (len(dish_name) > 2 and 
                        not dish_name.startswith('http') and
                        not dish_name.startswith('**') and
                        not dish_name.lower() in ['menu', 'about', 'hours', 'note']):
                        items.append(MenuItem(
                            dish_name=dish_name,
                            price=price,
                            category=current_category,
                            restaurant=restaurant.name,
                            cuisine=restaurant.cuisine,
                            price_range=restaurant.price_range,
                            address=restaurant.address
                        ))
                    break
        
        return items
    
    def _build_index(self) -> None:
        """Build search index"""
        for item in self.menu_items:
            # Index by full name
            key = item.dish_name.lower()
            if key not in self.dish_index:
                self.dish_index[key] = []
            self.dish_index[key].append(item)
            
            # Index by words
            for word in re.findall(r'\b\w{3,}\b', key):
                if word not in self.dish_index:
                    self.dish_index[word] = []
                if item not in self.dish_index[word]:
                    self.dish_index[word].append(item)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[MenuItem, float, str]]:
        """
        Search for dishes matching query.
        Returns: List of (MenuItem, score, match_type)
        """
        query = query.lower().strip()
        results = []
        seen = set()
        
        # 1. Exact match
        if query in self.dish_index:
            for item in self.dish_index[query]:
                if item.dish_name not in seen:
                    results.append((item, 1.0, "exact"))
                    seen.add(item.dish_name)
        
        # 2. Partial/substring match
        for item in self.menu_items:
            name_lower = item.dish_name.lower()
            if item.dish_name not in seen:
                if query in name_lower or name_lower in query:
                    results.append((item, 0.9, "partial"))
                    seen.add(item.dish_name)
        
        # 3. Fuzzy match using SequenceMatcher
        for item in self.menu_items:
            if item.dish_name not in seen:
                ratio = SequenceMatcher(None, query, item.dish_name.lower()).ratio()
                if ratio > 0.5:
                    results.append((item, ratio, "fuzzy"))
                    seen.add(item.dish_name)
        
        # 4. Keyword match
        query_words = set(re.findall(r'\b\w{3,}\b', query))
        for item in self.menu_items:
            if item.dish_name not in seen:
                item_words = set(re.findall(r'\b\w{3,}\b', item.dish_name.lower()))
                overlap = query_words & item_words
                if overlap:
                    score = len(overlap) / max(len(query_words), len(item_words))
                    if score > 0.3:
                        results.append((item, score * 0.8, "keyword"))
                        seen.add(item.dish_name)
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def find_dish(self, query: str) -> str:
        """Find dish and return formatted results"""
        results = self.search(query, top_k=5)
        
        if not results:
            return f"❌ No dishes found matching '{query}'"
        
        output = [f"\n🍜 Found {len(results)} result(s) for '{query}':\n"]
        output.append("=" * 55)
        
        for i, (item, score, match_type) in enumerate(results, 1):
            output.append(f"\n#{i} {item.dish_name}")
            if item.price:
                output.append(f"   💰 Price: {item.price}")
            output.append(f"   🏠 Restaurant: {item.restaurant}")
            output.append(f"   🍴 Cuisine: {item.cuisine}")
            if item.category:
                output.append(f"   📂 Category: {item.category}")
            if item.address:
                output.append(f"   📍 Address: {item.address}")
            output.append(f"   🎯 Match: {match_type} ({score:.0%})")
        
        return "\n".join(output)
    
    def list_all_dishes(self) -> str:
        """List all available dishes"""
        output = ["\n📋 All Available Dishes:\n"]
        
        for restaurant in self.restaurants:
            output.append(f"\n{'='*50}")
            output.append(f"🏠 {restaurant.name} ({restaurant.cuisine})")
            output.append(f"{'='*50}")
            
            for item in restaurant.menu_items:
                price_str = f" - {item.price}" if item.price else ""
                output.append(f"  • {item.dish_name}{price_str}")
        
        return "\n".join(output)


# Sample data for testing
SAMPLE_DATA = """# Asian Restaurants Frankfurt

## 1. Góc Phố - Vietnamese Street Food ⭐⭐⭐⭐

**Cuisine:** Vietnamese
**Price Range:** € (Budget-friendly)
**Address:** Schärfengäßchen 6, 60311 Frankfurt am Main

### Menu

**Nudelsuppen (Pho & Soups):**
- Pho Bo (Beef noodle soup) - 14€
- Pho Ga (Chicken noodle soup) - 13€
- Pho Chay (Vegetarian) - 12€
- Bun Bo Hue - 15€
- Bun Hai San (Seafood) - 18€

**Vorspeisen (Starters):**
- Cha Gio Re (Spring rolls) - 6€
- Cha Gio Ga (Chicken spring rolls) - 5€
- Goi Cuon Tom Thit (Summer rolls) - 6€
- Goi Cuon Chay (Vegetarian summer rolls) - 6€

**Reisgerichte (Rice Dishes):**
- Com Ga Nuong La Chanh - 14€
- Com Bui Saigon - 16€
- Com Ca Hoi Chien Mam Gung (Salmon) - 19€
- Com Tom Rim Nuoc Mam (Shrimp) - 17€

**Salate:**
- Goi Ga (Chicken salad) - 14€
- Goi Du Du Tom (Papaya shrimp) - 15€

---

## 2. Thong Thai ⭐⭐⭐⭐

**Cuisine:** Thai
**Price Range:** € (Budget-friendly)
**Address:** Meisengasse 12, 60313 Frankfurt

### Menu

**Hauptgerichte (Main Dishes):**
- Phad Thai - 7€
- Gai-Phad-Gra-Prau - 7€
- Pa-Naeng-Gai - 7€
- Kiow-Wan-Gai (Green Curry) - 7€
- Gaeng-Daeng-Gai (Red Curry) - 7€
- Massaman Curry - 8€
- Gai-Phad-Med Ma Muang (Cashew Chicken) - 7€

**Suppen (Soups):**
- Tom Yam Gai - 3€
- Tom Kha Pag - 3€
- Garnelensuppe - 4€

**Spezialitäten:**
- Ped-Thong-Thai (Duck) - 8€
- Gai-Thong-Thai (Chicken) - 8€
- Knusprig gebackene Ente - 10€

**Nudelgerichte:**
- Phad Sie Iew-Gai - 6€
- Bami-Phad-Gai - 6€

---

## 3. Zenzakan ⭐⭐⭐⭐

**Cuisine:** Pan-Asian (Japanese, Chinese)
**Price Range:** €€€ (Fine Dining)
**Address:** Taunusanlage 15, 60325 Frankfurt am Main

### Menu

**Sushi & Sashimi:**
- Salmon Sashimi - 18€
- Tuna Sashimi - 22€
- Dragon Roll - 16€
- Rainbow Roll - 18€

**From the Grill:**
- Wagyu Katsu Sando - 45€
- Black Pepper Beef - 28€
- Char Siu Chicken - 24€
- Lamb Chops - 32€

**Curries:**
- Thai Green Curry - 22€
- Massaman Beef - 26€

---

## 4. Pak Choi ⭐⭐⭐⭐

**Cuisine:** Chinese (Szechuan)
**Price Range:** €
**Address:** Dreieichstraße 7, 60594 Frankfurt

### Menu

**Szechuan Specialties:**
- Kung Pao Chicken - 12€
- Mapo Tofu - 10€
- Dan Dan Noodles - 9€
- Szechuan Dumplings - 8€
- Lamb with Cumin - 14€

**Soups:**
- Hot and Sour Soup - 6€
- Wonton Soup - 7€

**Noodles:**
- Chow Mein - 10€
- Singapore Noodles - 11€

---

## 5. China Restaurant Yung ⭐⭐⭐⭐⭐

**Cuisine:** Chinese (Cantonese)
**Price Range:** €€

### Menu

**Dim Sum:**
- Har Gow (Shrimp Dumplings) - 6€
- Siu Mai - 5€
- Char Siu Bao - 5€
- Spring Rolls - 4€

**Main Dishes:**
- Peking Duck - 38€
- Char Siu Pork - 10€
- Sweet and Sour Pork - 12€
- Braised Duck Wings - 9€

---

## 6. Kabuki Frankfurt ⭐⭐⭐⭐

**Cuisine:** Japanese
**Price Range:** €€€€
**Address:** Frankfurt Innenstadt

### Menu

**Sushi:**
- Omakase Sushi Set - 65€
- Chirashi Bowl - 28€
- Premium Nigiri Set - 45€

**Teppanyaki:**
- Wagyu Steak - 85€
- Lobster Teppanyaki - 55€
- Mixed Teppanyaki - 45€
"""


def main():
    """Interactive CLI"""
    print("=" * 60)
    print("🍜 Frankfurt Asian Restaurant Dish Finder")
    print("   Simple RAG - No Dependencies Required!")
    print("=" * 60)
    
    # Initialize RAG
    rag = SimpleRAG()
    
    # Try to load from file, otherwise use sample data
    md_paths = [
        "asian-restaurants-frankfurt-guide.md",
        "/mnt/user-data/outputs/asian-restaurants-frankfurt-guide.md",
        "restaurants.md"
    ]
    
    loaded = False
    for path in md_paths:
        if Path(path).exists():
            print(f"\n📂 Loading from: {path}")
            rag.load_from_markdown(path)
            loaded = True
            break
    
    if not loaded:
        print("\n📝 Using sample restaurant data...")
        rag.load_from_string(SAMPLE_DATA)
    
    # Interactive loop
    print("\n" + "=" * 60)
    print("Commands:")
    print("  - Type a dish name to search (e.g., 'pho', 'curry', 'sushi')")
    print("  - Type 'list' to see all dishes")
    print("  - Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n🔍 Search: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'list':
                print(rag.list_all_dishes())
                continue
            
            # Search
            print(rag.find_dish(query))
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
