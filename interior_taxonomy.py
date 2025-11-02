# ============================================
# COMPREHENSIVE INTERIOR DESIGN TAXONOMY
# For PlaybookTV Object Detection System
# ============================================

"""
This taxonomy includes 200+ interior design items categorized by:
- Furniture types & styles
- Decorative elements
- Lighting fixtures
- Textiles & soft furnishings
- Architectural features
- Electronics & appliances

Use this for YOLO object detection and SAM2 segmentation.
"""

# ============================================
# EXPANDED INTERIOR TAXONOMY
# ============================================

INTERIOR_TAXONOMY = {
    
    # ==========================================
    # SEATING FURNITURE (40+ types)
    # ==========================================
    "seating": {
        # Sofas & Couches
        "sectional_sofa": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room", "family_room"]},
        "loveseat": {"style_tags": ["traditional", "contemporary"], "typical_rooms": ["living_room", "bedroom"]},
        "chesterfield_sofa": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["living_room", "study"]},
        "sleeper_sofa": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room", "home_office"]},
        "settee": {"style_tags": ["traditional", "mid_century"], "typical_rooms": ["entryway", "bedroom"]},
        "chaise_lounge": {"style_tags": ["modern", "luxury"], "typical_rooms": ["bedroom", "living_room"]},
        
        # Chairs - Material-Specific
        "wicker_chair": {"style_tags": ["bohemian", "coastal"], "typical_rooms": ["patio", "sunroom"]},
        "rattan_chair": {"style_tags": ["bohemian", "tropical"], "typical_rooms": ["living_room", "patio"]},
        "leather_armchair": {"style_tags": ["traditional", "industrial"], "typical_rooms": ["living_room", "study"]},
        "velvet_armchair": {"style_tags": ["luxury", "art_deco"], "typical_rooms": ["living_room", "bedroom"]},
        
        # Chairs - Style-Specific
        "wingback_chair": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["living_room", "study"]},
        "club_chair": {"style_tags": ["traditional", "mid_century"], "typical_rooms": ["living_room", "study"]},
        "accent_chair": {"style_tags": ["modern", "eclectic"], "typical_rooms": ["living_room", "bedroom"]},
        "slipper_chair": {"style_tags": ["traditional", "contemporary"], "typical_rooms": ["bedroom", "dressing_room"]},
        "bergere_chair": {"style_tags": ["french_provincial", "traditional"], "typical_rooms": ["living_room", "bedroom"]},
        "barrel_chair": {"style_tags": ["mid_century", "contemporary"], "typical_rooms": ["living_room", "office"]},
        
        # Chairs - Functional
        "dining_chair": {"style_tags": ["any"], "typical_rooms": ["dining_room", "kitchen"]},
        "desk_chair": {"style_tags": ["modern", "industrial"], "typical_rooms": ["home_office", "study"]},
        "bar_stool": {"style_tags": ["modern", "industrial"], "typical_rooms": ["kitchen", "bar_area"]},
        "counter_stool": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["kitchen"]},
        "vanity_stool": {"style_tags": ["luxury", "traditional"], "typical_rooms": ["bedroom", "dressing_room"]},
        "piano_bench": {"style_tags": ["traditional", "mid_century"], "typical_rooms": ["living_room", "music_room"]},
        
        # Specialty Seating
        "ottoman": {"style_tags": ["any"], "typical_rooms": ["living_room", "bedroom"]},
        "pouf": {"style_tags": ["bohemian", "modern"], "typical_rooms": ["living_room", "bedroom"]},
        "bean_bag": {"style_tags": ["casual", "modern"], "typical_rooms": ["game_room", "kids_room"]},
        "rocking_chair": {"style_tags": ["traditional", "farmhouse"], "typical_rooms": ["nursery", "porch"]},
        "glider_chair": {"style_tags": ["modern", "traditional"], "typical_rooms": ["nursery", "living_room"]},
        "papasan_chair": {"style_tags": ["bohemian", "tropical"], "typical_rooms": ["living_room", "bedroom"]},
    },
    
    # ==========================================
    # TABLES (30+ types)
    # ==========================================
    "tables": {
        # Coffee Tables
        "coffee_table": {"style_tags": ["any"], "typical_rooms": ["living_room"]},
        "ottoman_coffee_table": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room"]},
        "nesting_coffee_tables": {"style_tags": ["modern", "scandinavian"], "typical_rooms": ["living_room"]},
        "trunk_coffee_table": {"style_tags": ["rustic", "vintage"], "typical_rooms": ["living_room"]},
        
        # Side Tables
        "side_table": {"style_tags": ["any"], "typical_rooms": ["living_room", "bedroom"]},
        "end_table": {"style_tags": ["any"], "typical_rooms": ["living_room"]},
        "corner_table": {"style_tags": ["any"], "typical_rooms": ["living_room", "bedroom"]},
        "nesting_side_tables": {"style_tags": ["modern", "scandinavian"], "typical_rooms": ["living_room"]},
        "c_table": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room"]},
        
        # Bedroom Tables
        "nightstand": {"style_tags": ["any"], "typical_rooms": ["bedroom"]},
        "bedside_table": {"style_tags": ["any"], "typical_rooms": ["bedroom"]},
        "vanity_table": {"style_tags": ["luxury", "traditional"], "typical_rooms": ["bedroom", "dressing_room"]},
        "makeup_vanity": {"style_tags": ["modern", "luxury"], "typical_rooms": ["bedroom"]},
        "dressing_table": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["bedroom"]},
        
        # Dining Tables
        "dining_table": {"style_tags": ["any"], "typical_rooms": ["dining_room", "kitchen"]},
        "extendable_dining_table": {"style_tags": ["modern", "traditional"], "typical_rooms": ["dining_room"]},
        "pedestal_dining_table": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["dining_room"]},
        "farmhouse_table": {"style_tags": ["farmhouse", "rustic"], "typical_rooms": ["dining_room", "kitchen"]},
        "breakfast_table": {"style_tags": ["any"], "typical_rooms": ["kitchen", "breakfast_nook"]},
        
        # Console & Entry Tables
        "console_table": {"style_tags": ["any"], "typical_rooms": ["entryway", "hallway", "living_room"]},
        "sofa_table": {"style_tags": ["any"], "typical_rooms": ["living_room"]},
        "entryway_table": {"style_tags": ["any"], "typical_rooms": ["entryway"]},
        "hall_table": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["hallway"]},
        
        # Work Tables
        "desk": {"style_tags": ["any"], "typical_rooms": ["home_office", "bedroom"]},
        "writing_desk": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["study", "bedroom"]},
        "computer_desk": {"style_tags": ["modern", "industrial"], "typical_rooms": ["home_office"]},
        "secretary_desk": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["study", "living_room"]},
        
        # Specialty Tables
        "bar_table": {"style_tags": ["modern", "industrial"], "typical_rooms": ["bar_area", "kitchen"]},
        "bar_cart": {"style_tags": ["mid_century", "art_deco"], "typical_rooms": ["living_room", "dining_room"]},
        "wine_table": {"style_tags": ["traditional", "modern"], "typical_rooms": ["dining_room", "wine_cellar"]},
        "accent_table": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "plant_stand": {"style_tags": ["bohemian", "modern"], "typical_rooms": ["any"]},
    },
    
    # ==========================================
    # STORAGE FURNITURE (35+ types)
    # ==========================================
    "storage": {
        # Bedroom Storage
        "dresser": {"style_tags": ["any"], "typical_rooms": ["bedroom"]},
        "chest_of_drawers": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["bedroom"]},
        "armoire": {"style_tags": ["traditional", "french_provincial"], "typical_rooms": ["bedroom"]},
        "wardrobe": {"style_tags": ["traditional", "scandinavian"], "typical_rooms": ["bedroom"]},
        "chifferobe": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["bedroom"]},
        "linen_chest": {"style_tags": ["traditional"], "typical_rooms": ["bedroom", "bathroom"]},
        
        # Shelving & Bookcases
        "bookshelf": {"style_tags": ["any"], "typical_rooms": ["living_room", "study", "bedroom"]},
        "bookcase": {"style_tags": ["any"], "typical_rooms": ["living_room", "study"]},
        "etagere": {"style_tags": ["traditional", "mid_century"], "typical_rooms": ["living_room", "dining_room"]},
        "ladder_shelf": {"style_tags": ["industrial", "modern"], "typical_rooms": ["living_room", "bedroom"]},
        "corner_shelf": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "floating_shelf": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["any"]},
        "wall_shelf": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "decorative_ladder": {"style_tags": ["farmhouse", "industrial"], "typical_rooms": ["bedroom", "bathroom"]},
        
        # Cabinets & Credenzas
        "credenza": {"style_tags": ["mid_century", "modern"], "typical_rooms": ["dining_room", "living_room"]},
        "sideboard": {"style_tags": ["traditional", "mid_century"], "typical_rooms": ["dining_room"]},
        "buffet": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["dining_room"]},
        "china_cabinet": {"style_tags": ["traditional"], "typical_rooms": ["dining_room"]},
        "hutch": {"style_tags": ["traditional", "farmhouse"], "typical_rooms": ["dining_room", "kitchen"]},
        "display_cabinet": {"style_tags": ["traditional", "modern"], "typical_rooms": ["living_room", "dining_room"]},
        "curio_cabinet": {"style_tags": ["traditional"], "typical_rooms": ["living_room"]},
        "linen_cabinet": {"style_tags": ["traditional"], "typical_rooms": ["bathroom", "hallway"]},
        
        # Media & Entertainment Storage
        "tv_stand": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room", "bedroom"]},
        "media_console": {"style_tags": ["modern", "mid_century"], "typical_rooms": ["living_room"]},
        "entertainment_center": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["living_room"]},
        
        # Entry & Hallway Storage
        "console_cabinet": {"style_tags": ["any"], "typical_rooms": ["entryway", "hallway"]},
        "hall_tree": {"style_tags": ["traditional", "farmhouse"], "typical_rooms": ["entryway"]},
        "coat_rack": {"style_tags": ["any"], "typical_rooms": ["entryway"]},
        "shoe_cabinet": {"style_tags": ["modern", "scandinavian"], "typical_rooms": ["entryway", "closet"]},
        
        # Specialty Storage
        "wine_rack": {"style_tags": ["modern", "industrial"], "typical_rooms": ["kitchen", "dining_room", "bar_area"]},
        "bar_cabinet": {"style_tags": ["mid_century", "art_deco"], "typical_rooms": ["dining_room", "living_room"]},
        "file_cabinet": {"style_tags": ["industrial", "modern"], "typical_rooms": ["home_office"]},
        "storage_ottoman": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room", "bedroom"]},
        "storage_bench": {"style_tags": ["any"], "typical_rooms": ["entryway", "bedroom"]},
        "trunk": {"style_tags": ["vintage", "rustic"], "typical_rooms": ["living_room", "bedroom"]},
        "chest": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["bedroom", "living_room"]},
    },
    
    # ==========================================
    # BEDS & BEDROOM FURNITURE (15+ types)
    # ==========================================
    "beds": {
        "king_bed": {"style_tags": ["any"], "typical_rooms": ["master_bedroom"]},
        "queen_bed": {"style_tags": ["any"], "typical_rooms": ["bedroom"]},
        "full_bed": {"style_tags": ["any"], "typical_rooms": ["bedroom", "guest_room"]},
        "twin_bed": {"style_tags": ["any"], "typical_rooms": ["kids_room", "guest_room"]},
        "bunk_bed": {"style_tags": ["modern", "traditional"], "typical_rooms": ["kids_room"]},
        "loft_bed": {"style_tags": ["modern", "industrial"], "typical_rooms": ["kids_room", "studio"]},
        "platform_bed": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["bedroom"]},
        "sleigh_bed": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["bedroom"]},
        "canopy_bed": {"style_tags": ["traditional", "romantic"], "typical_rooms": ["bedroom"]},
        "four_poster_bed": {"style_tags": ["traditional", "colonial"], "typical_rooms": ["bedroom"]},
        "upholstered_bed": {"style_tags": ["modern", "luxury"], "typical_rooms": ["bedroom"]},
        "murphy_bed": {"style_tags": ["modern"], "typical_rooms": ["studio", "guest_room"]},
        "daybed": {"style_tags": ["traditional", "modern"], "typical_rooms": ["guest_room", "home_office"]},
        "trundle_bed": {"style_tags": ["traditional", "contemporary"], "typical_rooms": ["kids_room", "guest_room"]},
        "futon": {"style_tags": ["modern", "casual"], "typical_rooms": ["guest_room", "home_office"]},
    },
    
    # ==========================================
    # LIGHTING FIXTURES (40+ types)
    # ==========================================
    "lighting": {
        # Ceiling Lights
        "chandelier": {"style_tags": ["traditional", "luxury", "art_deco"], "typical_rooms": ["dining_room", "entryway", "bedroom"]},
        "pendant_light": {"style_tags": ["modern", "industrial", "contemporary"], "typical_rooms": ["kitchen", "dining_room", "living_room"]},
        "hanging_light": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "drum_pendant": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["dining_room", "bedroom"]},
        "globe_pendant": {"style_tags": ["mid_century", "modern"], "typical_rooms": ["kitchen", "dining_room"]},
        "lantern_pendant": {"style_tags": ["traditional", "farmhouse"], "typical_rooms": ["entryway", "dining_room"]},
        "sputnik_chandelier": {"style_tags": ["mid_century", "atomic"], "typical_rooms": ["dining_room", "living_room"]},
        "crystal_chandelier": {"style_tags": ["traditional", "luxury"], "typical_rooms": ["dining_room", "entryway"]},
        "tiered_chandelier": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["entryway", "dining_room"]},
        "cage_pendant": {"style_tags": ["industrial", "farmhouse"], "typical_rooms": ["kitchen", "dining_room"]},
        
        # Ceiling Mounted
        "ceiling_light": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "flush_mount": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["bedroom", "hallway", "bathroom"]},
        "semi_flush_mount": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["bedroom", "hallway"]},
        "recessed_lighting": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["any"]},
        "track_lighting": {"style_tags": ["modern", "industrial"], "typical_rooms": ["kitchen", "art_gallery"]},
        "ceiling_fan_light": {"style_tags": ["any"], "typical_rooms": ["bedroom", "living_room"]},
        
        # Floor Lamps
        "floor_lamp": {"style_tags": ["any"], "typical_rooms": ["living_room", "bedroom", "home_office"]},
        "arc_floor_lamp": {"style_tags": ["mid_century", "modern"], "typical_rooms": ["living_room"]},
        "torchiere_lamp": {"style_tags": ["traditional", "contemporary"], "typical_rooms": ["living_room", "bedroom"]},
        "tripod_floor_lamp": {"style_tags": ["mid_century", "modern"], "typical_rooms": ["living_room", "bedroom"]},
        "pharmacy_lamp": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["home_office", "bedroom"]},
        "reading_lamp": {"style_tags": ["any"], "typical_rooms": ["bedroom", "living_room"]},
        
        # Table Lamps
        "table_lamp": {"style_tags": ["any"], "typical_rooms": ["living_room", "bedroom", "home_office"]},
        "desk_lamp": {"style_tags": ["modern", "industrial"], "typical_rooms": ["home_office", "study"]},
        "buffet_lamp": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["dining_room"]},
        "accent_lamp": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "candlestick_lamp": {"style_tags": ["traditional"], "typical_rooms": ["dining_room", "bedroom"]},
        "ginger_jar_lamp": {"style_tags": ["traditional", "chinoiserie"], "typical_rooms": ["bedroom", "living_room"]},
        
        # Wall Lights
        "wall_sconce": {"style_tags": ["any"], "typical_rooms": ["hallway", "bathroom", "bedroom"]},
        "picture_light": {"style_tags": ["traditional", "modern"], "typical_rooms": ["living_room", "hallway"]},
        "swing_arm_sconce": {"style_tags": ["industrial", "modern"], "typical_rooms": ["bedroom", "home_office"]},
        "candle_sconce": {"style_tags": ["traditional", "rustic"], "typical_rooms": ["dining_room", "hallway"]},
        "vanity_light": {"style_tags": ["any"], "typical_rooms": ["bathroom"]},
        
        # Specialty Lighting
        "string_lights": {"style_tags": ["bohemian", "modern"], "typical_rooms": ["bedroom", "patio"]},
        "edison_bulb_fixture": {"style_tags": ["industrial", "vintage"], "typical_rooms": ["dining_room", "kitchen"]},
        "neon_sign": {"style_tags": ["modern", "eclectic"], "typical_rooms": ["game_room", "bar_area"]},
        "led_strip_light": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["any"]},
        "under_cabinet_lighting": {"style_tags": ["modern"], "typical_rooms": ["kitchen"]},
        "cove_lighting": {"style_tags": ["modern", "luxury"], "typical_rooms": ["living_room", "bedroom"]},
    },
    
    # ==========================================
    # DECORATIVE ELEMENTS (50+ types)
    # ==========================================
    "decorative": {
        # Wall Decor
        "wall_art": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "framed_art": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "canvas_art": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room", "bedroom"]},
        "abstract_art": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room", "home_office"]},
        "landscape_painting": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["living_room", "dining_room"]},
        "portrait": {"style_tags": ["traditional", "classic"], "typical_rooms": ["living_room", "study"]},
        "photography_print": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["any"]},
        "tapestry": {"style_tags": ["bohemian", "eclectic"], "typical_rooms": ["bedroom", "living_room"]},
        "wall_hanging": {"style_tags": ["bohemian", "eclectic"], "typical_rooms": ["bedroom", "living_room"]},
        "macrame_wall_hanging": {"style_tags": ["bohemian", "coastal"], "typical_rooms": ["living_room", "bedroom"]},
        "mirror": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "decorative_mirror": {"style_tags": ["any"], "typical_rooms": ["entryway", "bathroom", "living_room"]},
        "round_mirror": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["bathroom", "bedroom"]},
        "sunburst_mirror": {"style_tags": ["mid_century", "art_deco"], "typical_rooms": ["living_room", "dining_room"]},
        "wall_clock": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "wall_shelf_decor": {"style_tags": ["any"], "typical_rooms": ["any"]},
        
        # Tabletop Decor
        "vase": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "flower_vase": {"style_tags": ["any"], "typical_rooms": ["dining_room", "living_room", "bedroom"]},
        "decorative_vase": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "ceramic_vase": {"style_tags": ["traditional", "contemporary"], "typical_rooms": ["living_room", "dining_room"]},
        "glass_vase": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["any"]},
        "decorative_bowl": {"style_tags": ["any"], "typical_rooms": ["dining_room", "coffee_table"]},
        "fruit_bowl": {"style_tags": ["any"], "typical_rooms": ["dining_room", "kitchen"]},
        "tray": {"style_tags": ["any"], "typical_rooms": ["coffee_table", "ottoman"]},
        "decorative_tray": {"style_tags": ["any"], "typical_rooms": ["coffee_table", "dresser"]},
        "candle": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "pillar_candle": {"style_tags": ["traditional", "contemporary"], "typical_rooms": ["dining_room", "living_room"]},
        "taper_candle": {"style_tags": ["traditional", "elegant"], "typical_rooms": ["dining_room"]},
        "decorative_candle": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "candle_holder": {"style_tags": ["any"], "typical_rooms": ["dining_room", "living_room"]},
        "candlestick": {"style_tags": ["traditional"], "typical_rooms": ["dining_room", "mantel"]},
        "hurricane_candle_holder": {"style_tags": ["traditional", "coastal"], "typical_rooms": ["dining_room", "patio"]},
        "lantern": {"style_tags": ["farmhouse", "coastal"], "typical_rooms": ["entryway", "patio"]},
        
        # Decorative Objects
        "sculpture": {"style_tags": ["modern", "traditional"], "typical_rooms": ["living_room", "home_office"]},
        "figurine": {"style_tags": ["traditional", "eclectic"], "typical_rooms": ["shelf", "mantel"]},
        "decorative_object": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "decorative_box": {"style_tags": ["any"], "typical_rooms": ["dresser", "coffee_table"]},
        "decorative_books": {"style_tags": ["any"], "typical_rooms": ["coffee_table", "bookshelf"]},
        "coffee_table_books": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room"]},
        "globe": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["home_office", "living_room"]},
        "hourglass": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["home_office", "shelf"]},
        
        # Plants & Nature
        "potted_plant": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "indoor_plant": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "succulent": {"style_tags": ["modern", "bohemian"], "typical_rooms": ["any"]},
        "fern": {"style_tags": ["traditional", "bohemian"], "typical_rooms": ["bathroom", "living_room"]},
        "snake_plant": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["bedroom", "living_room"]},
        "fiddle_leaf_fig": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room"]},
        "monstera": {"style_tags": ["tropical", "bohemian"], "typical_rooms": ["living_room", "bedroom"]},
        "fresh_flowers": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "dried_flowers": {"style_tags": ["bohemian", "farmhouse"], "typical_rooms": ["bedroom", "living_room"]},
        "flower_arrangement": {"style_tags": ["any"], "typical_rooms": ["dining_room", "entryway"]},
        "terrarium": {"style_tags": ["modern", "bohemian"], "typical_rooms": ["living_room", "home_office"]},
    },
    
    # ==========================================
    # TEXTILES & SOFT FURNISHINGS (30+ types)
    # ==========================================
    "textiles": {
        # Window Treatments
        "curtains": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "drapes": {"style_tags": ["traditional", "luxury"], "typical_rooms": ["living_room", "bedroom"]},
        "sheer_curtains": {"style_tags": ["modern", "romantic"], "typical_rooms": ["bedroom", "living_room"]},
        "blackout_curtains": {"style_tags": ["modern"], "typical_rooms": ["bedroom"]},
        "valance": {"style_tags": ["traditional"], "typical_rooms": ["kitchen", "bathroom"]},
        "blinds": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["any"]},
        "venetian_blinds": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["any"]},
        "roller_blinds": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["any"]},
        "roman_shades": {"style_tags": ["traditional", "contemporary"], "typical_rooms": ["bedroom", "living_room"]},
        "cellular_shades": {"style_tags": ["modern"], "typical_rooms": ["any"]},
        "shutters": {"style_tags": ["traditional", "farmhouse"], "typical_rooms": ["any"]},
        "plantation_shutters": {"style_tags": ["traditional", "colonial"], "typical_rooms": ["living_room", "bedroom"]},
        
        # Rugs & Floor Coverings
        "area_rug": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "persian_rug": {"style_tags": ["traditional", "luxury"], "typical_rooms": ["living_room", "dining_room"]},
        "oriental_rug": {"style_tags": ["traditional", "eclectic"], "typical_rooms": ["living_room", "bedroom"]},
        "moroccan_rug": {"style_tags": ["bohemian", "eclectic"], "typical_rooms": ["living_room", "bedroom"]},
        "shag_rug": {"style_tags": ["modern", "bohemian"], "typical_rooms": ["bedroom", "living_room"]},
        "jute_rug": {"style_tags": ["coastal", "bohemian"], "typical_rooms": ["living_room", "dining_room"]},
        "sisal_rug": {"style_tags": ["natural", "coastal"], "typical_rooms": ["any"]},
        "runner_rug": {"style_tags": ["any"], "typical_rooms": ["hallway", "kitchen"]},
        "round_rug": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["dining_room", "bedroom"]},
        "layered_rugs": {"style_tags": ["bohemian", "eclectic"], "typical_rooms": ["living_room"]},
        
        # Bedding & Pillows
        "throw_pillow": {"style_tags": ["any"], "typical_rooms": ["living_room", "bedroom"]},
        "decorative_pillow": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "accent_pillow": {"style_tags": ["any"], "typical_rooms": ["sofa", "bed"]},
        "lumbar_pillow": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room", "bedroom"]},
        "throw_blanket": {"style_tags": ["any"], "typical_rooms": ["living_room", "bedroom"]},
        "decorative_blanket": {"style_tags": ["any"], "typical_rooms": ["sofa", "bed"]},
        "bedding": {"style_tags": ["any"], "typical_rooms": ["bedroom"]},
        "duvet": {"style_tags": ["any"], "typical_rooms": ["bedroom"]},
        "comforter": {"style_tags": ["any"], "typical_rooms": ["bedroom"]},
        "bedspread": {"style_tags": ["traditional"], "typical_rooms": ["bedroom"]},
    },
    
    # ==========================================
    # ELECTRONICS & APPLIANCES (25+ types)
    # ==========================================
    "electronics": {
        # Entertainment
        "television": {"style_tags": ["modern"], "typical_rooms": ["living_room", "bedroom"]},
        "tv": {"style_tags": ["modern"], "typical_rooms": ["living_room", "bedroom"]},
        "flat_screen_tv": {"style_tags": ["modern"], "typical_rooms": ["living_room", "bedroom"]},
        "speakers": {"style_tags": ["modern"], "typical_rooms": ["living_room", "home_theater"]},
        "sound_system": {"style_tags": ["modern"], "typical_rooms": ["living_room"]},
        "soundbar": {"style_tags": ["modern"], "typical_rooms": ["living_room"]},
        "turntable": {"style_tags": ["vintage", "mid_century"], "typical_rooms": ["living_room"]},
        "record_player": {"style_tags": ["vintage", "mid_century"], "typical_rooms": ["living_room"]},
        "radio": {"style_tags": ["vintage", "modern"], "typical_rooms": ["any"]},
        "gaming_console": {"style_tags": ["modern"], "typical_rooms": ["living_room", "game_room"]},
        
        # Computing
        "computer": {"style_tags": ["modern"], "typical_rooms": ["home_office"]},
        "laptop": {"style_tags": ["modern"], "typical_rooms": ["home_office", "bedroom"]},
        "monitor": {"style_tags": ["modern"], "typical_rooms": ["home_office"]},
        "keyboard": {"style_tags": ["modern"], "typical_rooms": ["home_office"]},
        "mouse": {"style_tags": ["modern"], "typical_rooms": ["home_office"]},
        
        # Communication
        "phone": {"style_tags": ["modern"], "typical_rooms": ["any"]},
        "telephone": {"style_tags": ["traditional", "vintage"], "typical_rooms": ["home_office", "bedroom"]},
        "smartphone": {"style_tags": ["modern"], "typical_rooms": ["any"]},
        "tablet": {"style_tags": ["modern"], "typical_rooms": ["any"]},
        
        # Climate Control
        "air_conditioner": {"style_tags": ["modern"], "typical_rooms": ["any"]},
        "heater": {"style_tags": ["modern"], "typical_rooms": ["any"]},
        "fan": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "ceiling_fan": {"style_tags": ["traditional", "modern"], "typical_rooms": ["bedroom", "living_room"]},
        "humidifier": {"style_tags": ["modern"], "typical_rooms": ["bedroom"]},
        "air_purifier": {"style_tags": ["modern"], "typical_rooms": ["bedroom", "living_room"]},
    },
    
    # ==========================================
    # SPECIALTY FURNITURE (20+ types)
    # ==========================================
    "specialty": {
        # Utility Furniture
        "trolley": {"style_tags": ["industrial", "modern"], "typical_rooms": ["kitchen", "dining_room"]},
        "bar_cart": {"style_tags": ["mid_century", "art_deco"], "typical_rooms": ["living_room", "dining_room"]},
        "serving_cart": {"style_tags": ["traditional", "modern"], "typical_rooms": ["dining_room", "kitchen"]},
        "wine_bar": {"style_tags": ["modern", "luxury"], "typical_rooms": ["dining_room", "home_bar"]},
        "utility_cart": {"style_tags": ["industrial", "modern"], "typical_rooms": ["kitchen", "home_office"]},
        
        # Entryway Furniture
        "bench": {"style_tags": ["any"], "typical_rooms": ["entryway", "bedroom", "hallway"]},
        "entryway_bench": {"style_tags": ["any"], "typical_rooms": ["entryway"]},
        "mudroom_bench": {"style_tags": ["farmhouse", "modern"], "typical_rooms": ["mudroom", "entryway"]},
        "storage_bench": {"style_tags": ["any"], "typical_rooms": ["entryway", "bedroom"]},
        
        # Room Dividers
        "room_divider": {"style_tags": ["any"], "typical_rooms": ["studio", "living_room"]},
        "screen": {"style_tags": ["traditional", "asian"], "typical_rooms": ["living_room", "bedroom"]},
        "folding_screen": {"style_tags": ["traditional", "eclectic"], "typical_rooms": ["bedroom", "living_room"]},
        "bookcase_divider": {"style_tags": ["modern", "industrial"], "typical_rooms": ["studio", "living_room"]},
        
        # Specialty Items
        "fireplace": {"style_tags": ["traditional", "modern"], "typical_rooms": ["living_room"]},
        "fireplace_screen": {"style_tags": ["traditional"], "typical_rooms": ["living_room"]},
        "fireplace_tools": {"style_tags": ["traditional", "rustic"], "typical_rooms": ["living_room"]},
        "fireplace_mantel": {"style_tags": ["traditional", "farmhouse"], "typical_rooms": ["living_room"]},
        "wine_rack": {"style_tags": ["modern", "industrial"], "typical_rooms": ["kitchen", "dining_room"]},
        "magazine_rack": {"style_tags": ["mid_century", "modern"], "typical_rooms": ["living_room", "bathroom"]},
        "umbrella_stand": {"style_tags": ["traditional", "modern"], "typical_rooms": ["entryway"]},
    },
    
    # ==========================================
    # ARCHITECTURAL FEATURES (15+ types)
    # ==========================================
    "architectural": {
        # Flooring
        "hardwood_floor": {"style_tags": ["traditional", "contemporary"], "typical_rooms": ["any"]},
        "tile_floor": {"style_tags": ["any"], "typical_rooms": ["kitchen", "bathroom"]},
        "marble_floor": {"style_tags": ["luxury", "traditional"], "typical_rooms": ["entryway", "bathroom"]},
        "concrete_floor": {"style_tags": ["industrial", "modern"], "typical_rooms": ["any"]},
        "carpet": {"style_tags": ["traditional", "modern"], "typical_rooms": ["bedroom", "living_room"]},
        "laminate_floor": {"style_tags": ["modern"], "typical_rooms": ["any"]},
        
        # Windows
        "window": {"style_tags": ["any"], "typical_rooms": ["any"]},
        "bay_window": {"style_tags": ["traditional"], "typical_rooms": ["living_room", "bedroom"]},
        "french_window": {"style_tags": ["traditional", "french_provincial"], "typical_rooms": ["living_room"]},
        "casement_window": {"style_tags": ["traditional", "contemporary"], "typical_rooms": ["any"]},
        "picture_window": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["living_room"]},
        "skylight": {"style_tags": ["modern"], "typical_rooms": ["bedroom", "bathroom"]},
        
        # Walls & Ceiling
        "accent_wall": {"style_tags": ["modern", "contemporary"], "typical_rooms": ["any"]},
        "wainscoting": {"style_tags": ["traditional", "transitional"], "typical_rooms": ["dining_room", "hallway"]},
        "crown_molding": {"style_tags": ["traditional"], "typical_rooms": ["any"]},
    },
}

# ============================================
# EXPORT FUNCTIONS
# ============================================

def get_all_categories():
    """Return flat list of all item categories"""
    all_items = []
    for category_type, items in INTERIOR_TAXONOMY.items():
        all_items.extend(items.keys())
    return all_items

def get_items_by_room(room_type):
    """Get all items typically found in a specific room"""
    items_in_room = []
    for category_type, items in INTERIOR_TAXONOMY.items():
        for item_name, item_data in items.items():
            if room_type in item_data.get("typical_rooms", []):
                items_in_room.append(item_name)
    return items_in_room

def get_items_by_style(style):
    """Get all items typically associated with a design style"""
    items_with_style = []
    for category_type, items in INTERIOR_TAXONOMY.items():
        for item_name, item_data in items.items():
            if style in item_data.get("style_tags", []):
                items_with_style.append(item_name)
    return items_with_style

def get_category_type(item_name):
    """Return the category type for a given item"""
    for category_type, items in INTERIOR_TAXONOMY.items():
        if item_name in items:
            return category_type
    return None

def get_taxonomy_stats():
    """Return statistics about the taxonomy"""
    total_items = sum(len(items) for items in INTERIOR_TAXONOMY.values())
    stats = {
        'total_categories': len(INTERIOR_TAXONOMY),
        'total_items': total_items,
        'items_per_category': {cat: len(items) for cat, items in INTERIOR_TAXONOMY.items()}
    }
    return stats

# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == "__main__":
    # Print statistics
    stats = get_taxonomy_stats()
    print(f"\n📊 Interior Design Taxonomy Statistics")
    print("=" * 50)
    print(f"Total Categories: {stats['total_categories']}")
    print(f"Total Items: {stats['total_items']}")
    print("\nItems per category:")
    for category, count in stats['items_per_category'].items():
        print(f"  {category:20s} {count:3} items")
    
    # Example queries
    print("\n" + "=" * 50)
    print("Example Queries:")
    print("=" * 50)
    
    living_room_items = get_items_by_room("living_room")
    print(f"\nItems in living room: {len(living_room_items)}")
    print(f"Examples: {', '.join(living_room_items[:10])}...")
    
    modern_items = get_items_by_style("modern")
    print(f"\nModern style items: {len(modern_items)}")
    print(f"Examples: {', '.join(modern_items[:10])}...")
    
    all_categories = get_all_categories()
    print(f"\nTotal unique items: {len(all_categories)}")
