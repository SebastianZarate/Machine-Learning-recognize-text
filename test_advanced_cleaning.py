"""Test advanced text cleaning with URLs, emails, mentions, numbers, etc."""
from src.model import clean_text

print("=" * 80)
print("PRUEBAS DE LIMPIEZA AVANZADA DE TEXTO")
print("=" * 80 + "\n")

# Test cases con URLs, emails, menciones, hashtags, números, repeticiones
test_cases = [
    # URLs
    ("Check out http://example.com for more info", "URL http"),
    ("Visit https://movie-reviews.com/best", "URL https"),
    ("Go to www.imdb.com for ratings", "URL www"),
    
    # Emails
    ("Contact me at john@example.com for details", "Email básico"),
    ("Send feedback to support@movie-reviews.org", "Email en dominio"),
    
    # Menciones y hashtags
    ("@john said the movie was #awesome!", "Mención + hashtag"),
    ("Great film! #mustwatch #cinema #2024", "Múltiples hashtags"),
    ("RT @reviewer: Best movie ever! #recommended", "RT con mención"),
    
    # Números (con diferentes estrategias)
    ("This movie from 2024 got a 10/10 rating", "Años y calificaciones"),
    ("The sequel in 1999 was rated 8.5 stars", "Año y decimal"),
    
    # Caracteres repetidos
    ("Sooooo good! I loooooved it!", "Repeticiones múltiples"),
    ("Wooooow! Amaaazing performance!!!", "Repeticiones extremas"),
    
    # Casos mixtos complejos
    ("Email me at user@domain.com or check https://site.com #awesome", "Todo mixto"),
    ("@user posted: Movie rated 9/10! Visit www.example.com #film", "Caso complejo"),
    ("Sooooo bored... 2/10 rating. Contact: help@support.com", "Repeticiones + números + email"),
]

print("📝 Pruebas de limpieza con diferentes estrategias de números\n")
print("=" * 80 + "\n")

for text, category in test_cases:
    print(f"📌 {category}:")
    print(f"   Original: '{text}'")
    
    # Probar las 3 estrategias de números
    remove = clean_text(text, normalize_numbers='remove')
    token = clean_text(text, normalize_numbers='token')
    keep = clean_text(text, normalize_numbers='keep')
    
    print(f"\n   ✓ Con normalize_numbers='remove':")
    print(f"     '{remove}'")
    
    print(f"\n   ✓ Con normalize_numbers='token':")
    print(f"     '{token}'")
    
    print(f"\n   ✓ Con normalize_numbers='keep':")
    print(f"     '{keep}'")
    
    print()

print("=" * 80)
print("✅ LIMPIEZA AVANZADA COMPLETADA")
print("=" * 80)

# Verificaciones específicas
print("\n🔍 Verificaciones específicas:\n")

# Test 1: URLs eliminadas
test1 = clean_text("Visit http://example.com now", return_tokens=True)
print(f"✓ URLs: 'Visit http://example.com now' → {test1}")
assert 'http' not in ' '.join(test1) and 'example' not in ' '.join(test1), "URLs deben eliminarse"

# Test 2: Emails eliminados
test2 = clean_text("Email user@domain.com for help", return_tokens=True)
print(f"✓ Emails: 'Email user@domain.com for help' → {test2}")
assert 'domain' not in ' '.join(test2) and '@' not in ' '.join(test2), "Emails deben eliminarse"

# Test 3: Menciones eliminadas, hashtags preservados
test3 = clean_text("@user said #awesome movie", return_tokens=True)
print(f"✓ Menciones/Hashtags: '@user said #awesome movie' → {test3}")
assert 'user' not in test3 and 'awesome' in test3, "Menciones eliminadas, hashtags preservados"

# Test 4: Números según estrategia
test4_remove = clean_text("Movie from 2024 rated 10", return_tokens=True, normalize_numbers='remove')
test4_token = clean_text("Movie from 2024 rated 10", return_tokens=True, normalize_numbers='token')
test4_keep = clean_text("Movie from 2024 rated 10", return_tokens=True, normalize_numbers='keep')

print(f"\n✓ Números (remove): {test4_remove}")
print(f"✓ Números (token): {test4_token}")
print(f"✓ Números (keep): {test4_keep}")

assert '2024' not in test4_remove and '10' not in test4_remove, "remove debe eliminar números"
assert '<NUM>' in test4_token or 'num' in test4_token, "token debe generar <NUM> o num (lemmatizado)"
assert '2024' in test4_keep or '10' in test4_keep, "keep debe mantener números"

# Test 5: Repeticiones normalizadas
test5 = clean_text("Sooooo goooood", return_tokens=True, normalize_numbers='remove')
print(f"\n✓ Repeticiones: 'Sooooo goooood' → {test5}")
# Debe convertir sooooo → soo, goooood → good

# Test 6: Caso complejo real
complex_text = """
@reviewer posted: Check out https://movie-review.com! 
The 2024 film got 10/10 rating. Sooooo amazing! #mustwatch
Email me at contact@reviews.com for more.
"""
complex_clean = clean_text(complex_text, normalize_numbers='token')
print(f"\n✓ Caso complejo:")
print(f"   Original: {complex_text.strip()}")
print(f"   Limpio: '{complex_clean}'")

print("\n" + "=" * 80)
print("📊 COMPARACIÓN DE ESTRATEGIAS DE NÚMEROS")
print("=" * 80 + "\n")

number_test_cases = [
    "The 2024 movie got 10/10 rating",
    "Films from the 1950s are classics",
    "Rated 8.5 out of 10 stars",
]

print("Estrategia recomendada según caso de uso:\n")

for text in number_test_cases:
    remove = clean_text(text, normalize_numbers='remove')
    token = clean_text(text, normalize_numbers='token')
    keep = clean_text(text, normalize_numbers='keep')
    
    print(f"Original: '{text}'")
    print(f"  • remove: '{remove}' (menor vocabulario, pierde info temporal)")
    print(f"  • token:  '{token}' (balance, mantiene señal numérica)")
    print(f"  • keep:   '{keep}' (máxima info, mayor vocabulario)")
    print()

print("💡 Recomendaciones:")
print("   • 'remove': Para textos sin info temporal/numérica relevante")
print("   • 'token': RECOMENDADO - Balance entre info y vocabulario")
print("   • 'keep': Solo si años/ratings son features importantes")

print("\n✅ Todas las verificaciones completadas!")
