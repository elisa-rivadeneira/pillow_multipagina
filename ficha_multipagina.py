from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
import io
import logging
import re
from datetime import datetime
from typing import List, Tuple
import math
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class CombinarDocumentosRequest(BaseModel):
    rutas_archivos: List[str]

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def sanitize_filename(text: str) -> str:
    """Sanitiza un string para usarlo como nombre de archivo."""
    if not text:
        return "Sin_Titulo"
    sanitized = re.sub(r'[^\w\s-]', '', text)
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = sanitized[:50]
    return sanitized if sanitized else "Sin_Titulo"

def to_title_case(text: str) -> str:
    """Convierte a Title Case."""
    if not text:
        return ""
    minor_words = [
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'de', 'a', 'en', 'por', 'con', 'sin', 'sobre', 'tras',
        'y', 'o', 'ni', 'pero', 'mas', 'que'
    ]
    words = text.lower().split()
    title_cased_words = []
    
    for i, word in enumerate(words):
        if i == 0 or i == len(words) - 1:
            title_cased_words.append(word.capitalize())
        elif word not in minor_words:
            title_cased_words.append(word.capitalize())
        else:
            title_cased_words.append(word)
    return " ".join(title_cased_words)

def parse_markdown_line(line):
    """Parsea markdown en una línea"""
    segments = []
    pattern = r'(\*\*\*[^\*]+\*\*\*|\*\*[^\*]+\*\*|\*[^\*]+\*)'
    
    last_end = 0
    for match in re.finditer(pattern, line):
        if match.start() > last_end:
            segments.append((line[last_end:match.start()], 'normal'))
        
        matched_text = match.group(0)
        if matched_text.startswith('***') and matched_text.endswith('***'):
            segments.append((matched_text[3:-3], 'bold'))
        elif matched_text.startswith('**') and matched_text.endswith('**'):
            segments.append((matched_text[2:-2], 'bold'))
        elif matched_text.startswith('*') and matched_text.endswith('*'):
            segments.append((matched_text[1:-1], 'bold'))
        
        last_end = match.end()
    
    if last_end < len(line):
        segments.append((line[last_end:], 'normal'))
    
    return segments if segments else [(line, 'normal')]

def draw_formatted_line(draw, x, y, line, fonts, color, max_width_px=None):
    """Dibuja línea con markdown y justificación opcional."""
    segments = parse_markdown_line(line)
    current_x = x
    
    extra_space_per_gap = 0
    
    if max_width_px:
        total_text_width_with_default_spaces = 0
        num_spaces = 0
        
        for seg_text, seg_style in segments:
            font = fonts.get(seg_style, fonts['normal'])
            try:
                text_width = draw.textlength(seg_text, font=font)
                total_text_width_with_default_spaces += text_width
            except AttributeError:
                bbox = draw.textbbox((0, 0), seg_text, font=font)
                total_text_width_with_default_spaces += bbox[2] - bbox[0]
            num_spaces += seg_text.count(' ')
        
        if num_spaces > 0 and (total_text_width_with_default_spaces / max_width_px > 0.7):
            remaining_width = max_width_px - total_text_width_with_default_spaces
            extra_space_per_gap = remaining_width / num_spaces
    
    for text, style in segments:
        font = fonts.get(style, fonts['normal'])
        draw.text((current_x, y), text, font=font, fill=color)
        
        try:
            text_width = draw.textlength(text, font=font)
        except AttributeError:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
        
        current_x += text_width
        
        if extra_space_per_gap > 0:
            spaces_in_segment = text.count(' ')
            current_x += spaces_in_segment * extra_space_per_gap
            
    return max_width_px if extra_space_per_gap > 0 else (current_x - x)

def wrap_text_with_markdown(text, fonts, max_width_px, draw):
    """Divide texto respetando markdown Y PÁRRAFOS"""
    text_normalized = re.sub(r'\n{3,}', '\n\n', text)
    paragraphs = text_normalized.split('\n\n')
    
    all_lines = []
    
    for para_idx, para in enumerate(paragraphs):
        if not para.strip():
            continue
        
        para_lines = para.split('\n')
        
        for line_idx, line in enumerate(para_lines):
            if not line.strip():
                continue
            
            words = line.strip().split()
            current_line_words = []
            
            for word in words:
                test_line = ' '.join(current_line_words + [word])
                segments = parse_markdown_line(test_line)
                
                total_width = 0
                for seg_text, seg_style in segments:
                    font = fonts.get(seg_style, fonts['normal'])
                    try:
                        bbox = draw.textbbox((0, 0), seg_text, font=font)
                    except Exception:
                        bbox = (0, 0, len(seg_text) * 20, 0) 
                    total_width += bbox[2] - bbox[0]
                
                if total_width <= max_width_px:
                    current_line_words.append(word)
                else:
                    if current_line_words:
                        all_lines.append((' '.join(current_line_words), 'text'))
                    current_line_words = [word]
            
            if current_line_words:
                all_lines.append((' '.join(current_line_words), 'text'))
        
        if para_idx < len(paragraphs) - 1:
            all_lines.append(('', 'paragraph_break'))
    
    return all_lines

def draw_wavy_border(draw, a4_width, a4_height):
    """Dibuja borde ondulado infantil"""
    import math
    colors = ['#FF6B9D', '#FFA07A', '#FFD93D', '#6BCF7F', '#4ECDC4', '#95E1D3']
    margin = 60
    wave_width = 40

    for x in range(margin, a4_width - margin, 10):
        wave_y_top = margin + wave_width * math.sin(x * 0.05)
        draw.ellipse([x, wave_y_top - 5, x + 10, wave_y_top + 5], fill=colors[x % len(colors)])

    for x in range(margin, a4_width - margin, 10):
        wave_y_bottom = a4_height - margin - wave_width * math.sin(x * 0.05)
        draw.ellipse([x, wave_y_bottom - 5, x + 10, wave_y_bottom + 5], fill=colors[x % len(colors)])

def detectar_espacios_libres(img: Image.Image) -> list:
    """🔍 Detecta zonas de la imagen con menos detalle para posicionar personajes (versión optimizada PIL)."""
    from PIL import ImageFilter, ImageStat

    # Detectar bordes para encontrar zonas con menos detalle
    edges = img.filter(ImageFilter.EDGE_ENHANCE_MORE).convert('L')

    # Definir cuadrantes estratégicos
    w, h = img.size
    cuadrantes = {
        'superior_izq': (0, 0, w//2, h//2),
        'superior_der': (w//2, 0, w, h//2),
        'inferior_izq': (0, h//2, w//2, h),
        'inferior_der': (w//2, h//2, w, h),
        'centro': (w//4, h//4, 3*w//4, 3*h//4)
    }

    espacios_libres = []
    for zona, (x1, y1, x2, y2) in cuadrantes.items():
        # Recortar región y calcular densidad de bordes
        region = edges.crop((x1, y1, x2, y2))
        stats = ImageStat.Stat(region)
        densidad = stats.mean[0]  # Promedio de intensidad (0-255)
        espacios_libres.append((zona, densidad, (x1, y1, x2, y2)))

    # Ordenar por menos densidad (valores más bajos = menos detalle = mejor para personaje)
    return sorted(espacios_libres, key=lambda x: x[1])

def remover_fondo_blanco(img: Image.Image, umbral: int = 240) -> Image.Image:
    """🎭 Remueve fondo blanco/claro del personaje para hacerlo transparente."""
    # Convertir a RGBA si no lo está
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Obtener datos de píxeles
    data = img.getdata()
    nueva_data = []

    for item in data:
        # Si el píxel es muy blanco (R, G, B altos), hacerlo transparente
        if item[0] > umbral and item[1] > umbral and item[2] > umbral:
            # Hacer transparente (alpha = 0)
            nueva_data.append((item[0], item[1], item[2], 0))
        else:
            # Mantener píxel original
            nueva_data.append(item)

    # Aplicar nueva data
    img_transparente = Image.new('RGBA', img.size)
    img_transparente.putdata(nueva_data)

    return img_transparente

def aplicar_efectos_visuales(img: Image.Image, personaje_pos: tuple, personaje_size: tuple) -> Image.Image:
    """Aplica efectos visuales espectaculares."""
    from PIL import ImageFilter, ImageEnhance

    # Crear sombra del personaje
    sombra = Image.new('RGBA', img.size, (0, 0, 0, 0))
    sombra_draw = ImageDraw.Draw(sombra)

    x, y = personaje_pos
    w, h = personaje_size

    # Sombra proyectada (offset diagonal)
    shadow_offset_x, shadow_offset_y = 15, 20
    shadow_x = x + shadow_offset_x
    shadow_y = y + shadow_offset_y

    # Crear forma de sombra elíptica
    sombra_draw.ellipse([
        shadow_x, shadow_y + h//2,
        shadow_x + w, shadow_y + h + 30
    ], fill=(0, 0, 0, 80))

    # Desenfocar sombra
    sombra = sombra.filter(ImageFilter.GaussianBlur(radius=8))

    # Aplicar sombra al fondo
    img = Image.alpha_composite(img.convert('RGBA'), sombra)

    # Añadir viñeta sutil
    viñeta = Image.new('RGBA', img.size, (0, 0, 0, 0))
    viñeta_draw = ImageDraw.Draw(viñeta)

    # Gradiente radial simulado con círculos concéntricos
    center_x, center_y = img.width // 2, img.height // 2
    max_radius = max(img.width, img.height) // 2

    for i in range(0, max_radius, 20):
        alpha = int(30 * (i / max_radius))  # Viñeta muy sutil
        viñeta_draw.ellipse([
            center_x - max_radius + i, center_y - max_radius + i,
            center_x + max_radius - i, center_y + max_radius - i
        ], outline=(0, 0, 0, alpha), width=15)

    img = Image.alpha_composite(img, viñeta)

    return img.convert('RGB')

def get_layout_dinamico(numero_pagina: int, total_paginas: int) -> dict:
    """Determina el layout según la página para crear variedad visual."""
    layouts = [
        # Página 1: Presentación épica
        {
            'personaje_scale': 0.6, 'position': 'centro_derecha',
            'efectos': ['sombra_dramática', 'brillo'], 'tipo': 'presentacion'
        },
        # Página 2: Acción/Movimiento
        {
            'personaje_scale': 0.45, 'position': 'esquina_inferior_izq',
            'efectos': ['dinamismo'], 'tipo': 'accion'
        },
        # Página 3: Exploración
        {
            'personaje_scale': 0.3, 'position': 'superior_izquierda',
            'efectos': ['aventura'], 'tipo': 'exploracion'
        },
        # Página 4: Conflicto/Tensión
        {
            'personaje_scale': 0.5, 'position': 'centro_izquierda',
            'efectos': ['tension'], 'tipo': 'conflicto'
        },
        # Página 5+: Resolución
        {
            'personaje_scale': 0.55, 'position': 'centro',
            'efectos': ['triunfo'], 'tipo': 'resolucion'
        }
    ]

    # Ciclar layouts si hay más de 5 páginas
    return layouts[(numero_pagina - 1) % len(layouts)]

def combinar_fondo_personaje(fondo_img: Image.Image, personaje_img: Image.Image, header_height: int, numero_pagina: int = 1, total_paginas: int = 1) -> Image.Image:
    """🎨 COMPOSICIÓN ÉPICA ESTILO PIXAR - Combina fondo + personaje con efectos visuales espectaculares."""
    a4_width = 2480

    # ============ 1. PREPARAR FONDO ============
    target_aspect = a4_width / header_height
    fondo_aspect = fondo_img.width / fondo_img.height

    if fondo_aspect < target_aspect:
        new_width = a4_width
        new_height = int(a4_width / fondo_aspect)
        fondo_resized = fondo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        top_crop = max(0, (new_height - header_height) // 2)
        fondo_final = fondo_resized.crop((0, top_crop, new_width, top_crop + header_height))
    else:
        new_height = header_height
        new_width = int(header_height * fondo_aspect)
        fondo_resized = fondo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left_crop = max(0, (new_width - a4_width) // 2)
        fondo_final = fondo_resized.crop((left_crop, 0, left_crop + a4_width, new_height))

    # ============ 2. LAYOUT DINÁMICO ============
    layout = get_layout_dinamico(numero_pagina, total_paginas)

    # ============ 3. DETECTAR ESPACIOS LIBRES ============
    espacios_libres = detectar_espacios_libres(fondo_final)
    mejor_zona = espacios_libres[0][0] if espacios_libres else 'superior_der'

    # ============ 4. REDIMENSIONAR PERSONAJE ============
    personaje_scale = layout['personaje_scale']
    max_personaje_width = int(a4_width * personaje_scale)
    personaje_aspect = personaje_img.width / personaje_img.height

    if personaje_img.width > max_personaje_width:
        personaje_width = max_personaje_width
        personaje_height = int(max_personaje_width / personaje_aspect)
    else:
        personaje_width = int(personaje_img.width * personaje_scale)
        personaje_height = int(personaje_img.height * personaje_scale)

    # Límite de altura
    if personaje_height > header_height * 0.85:
        personaje_height = int(header_height * 0.85)
        personaje_width = int(personaje_height * personaje_aspect)

    personaje_resized = personaje_img.resize((personaje_width, personaje_height), Image.Resampling.LANCZOS)

    # ============ 5. POSICIONAMIENTO INTELIGENTE ============
    position_type = layout['position']

    if position_type == 'centro_derecha':
        personaje_x = a4_width - personaje_width - 120
        personaje_y = (header_height - personaje_height) // 2
    elif position_type == 'esquina_inferior_izq':
        personaje_x = 80
        personaje_y = header_height - personaje_height - 60
    elif position_type == 'superior_izquierda':
        personaje_x = 100
        personaje_y = 80
    elif position_type == 'centro_izquierda':
        personaje_x = 150
        personaje_y = (header_height - personaje_height) // 2
    elif position_type == 'centro':
        personaje_x = (a4_width - personaje_width) // 2
        personaje_y = (header_height - personaje_height) // 2
    else:
        # Fallback: usar zona con menos detalle
        if mejor_zona == 'superior_izq':
            personaje_x, personaje_y = 100, 80
        elif mejor_zona == 'superior_der':
            personaje_x = a4_width - personaje_width - 120
            personaje_y = 80
        elif mejor_zona == 'inferior_izq':
            personaje_x, personaje_y = 80, header_height - personaje_height - 60
        else:  # inferior_der o centro
            personaje_x = a4_width - personaje_width - 120
            personaje_y = header_height - personaje_height - 60

    # ============ 6. APLICAR EFECTOS VISUALES ============
    resultado = fondo_final.copy().convert('RGBA')
    resultado = aplicar_efectos_visuales(resultado, (personaje_x, personaje_y), (personaje_width, personaje_height))

    # ============ 7. COMPOSICIÓN FINAL ============
    resultado = resultado.convert('RGBA')

    # Pegar personaje con transparencia
    if personaje_resized.mode == 'RGBA':
        resultado.paste(personaje_resized, (personaje_x, personaje_y), personaje_resized)
    else:
        resultado.paste(personaje_resized, (personaje_x, personaje_y))

    # ============ 8. EFECTOS FINALES ============
    # Realce de color sutil
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Color(resultado)
    resultado = enhancer.enhance(1.1)  # +10% saturación

    enhancer = ImageEnhance.Contrast(resultado)
    resultado = enhancer.enhance(1.05)  # +5% contraste

    logger.info(f"🎨 Composición creada - Página {numero_pagina}: Layout '{layout['tipo']}', Zona '{mejor_zona}'")

    return resultado.convert('RGB')

def crear_fondo_completo_epico(fondo_img: Image.Image, personaje_img: Image.Image, a4_width: int, a4_height: int, numero_pagina: int = 1) -> Image.Image:
    """🎬 ESTILO FONDO COMPLETO - Fondo ocupa toda la página, personaje grande estilo WOW cinematográfico."""

    logger.info(f"🎬 Creando fondo completo: {a4_width}x{a4_height}")

    # ============ 1. FONDO GIGANTE TODA LA PÁGINA A4 COMPLETA ============
    fondo_aspect = fondo_img.width / fondo_img.height
    page_aspect = a4_width / a4_height

    if fondo_aspect < page_aspect:
        # Imagen más alta que la página - ajustar por ancho
        new_width = a4_width
        new_height = int(a4_width / fondo_aspect)
        fondo_resized = fondo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Centrar verticalmente
        y_offset = (new_height - a4_height) // 2
        fondo_final = fondo_resized.crop((0, y_offset, a4_width, y_offset + a4_height))
    else:
        # Imagen más ancha que la página - ajustar por altura
        new_height = a4_height
        new_width = int(a4_height * fondo_aspect)
        fondo_resized = fondo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Centrar horizontalmente
        x_offset = (new_width - a4_width) // 2
        fondo_final = fondo_resized.crop((x_offset, 0, x_offset + a4_width, a4_height))

    # Verificar que el fondo final sea exactamente A4
    if fondo_final.size != (a4_width, a4_height):
        logger.warning(f"⚠️ Redimensionando fondo a A4: {fondo_final.size} → {a4_width}x{a4_height}")
        fondo_final = fondo_final.resize((a4_width, a4_height), Image.Resampling.LANCZOS)

    # ============ 2. POSICIONAMIENTO DINÁMICO DEL PERSONAJE ============
    personaje_scales = [0.7, 0.6, 0.65, 0.75, 0.8]  # Escalas grandes épicas
    personaje_positions = [
        'derecha_dramatico',    # Página 1: Presentación épica
        'izquierda_accion',     # Página 2: Acción dinámica
        'centro_exploracion',   # Página 3: Exploración
        'derecha_tension',      # Página 4: Tensión
        'centro_triunfo'        # Página 5: Resolución triunfante
    ]

    scale = personaje_scales[(numero_pagina - 1) % len(personaje_scales)]
    position_type = personaje_positions[(numero_pagina - 1) % len(personaje_positions)]

    # Calcular tamaño del personaje (GRANDE estilo WOW)
    max_personaje_height = int(a4_height * scale)
    personaje_aspect = personaje_img.width / personaje_img.height
    personaje_height = min(max_personaje_height, personaje_img.height)
    personaje_width = int(personaje_height * personaje_aspect)

    # Limitar ancho si es necesario
    if personaje_width > a4_width * 0.6:
        personaje_width = int(a4_width * 0.6)
        personaje_height = int(personaje_width / personaje_aspect)

    personaje_resized = personaje_img.resize((personaje_width, personaje_height), Image.Resampling.LANCZOS)

    # ============ 3. POSICIONAMIENTO SEGÚN TIPO ============
    if position_type == 'derecha_dramatico':
        personaje_x = a4_width - personaje_width - 100
        personaje_y = a4_height - personaje_height - 100
    elif position_type == 'izquierda_accion':
        personaje_x = 100
        personaje_y = a4_height - personaje_height - 150
    elif position_type == 'centro_exploracion':
        personaje_x = (a4_width - personaje_width) // 2
        personaje_y = a4_height - personaje_height - 200
    elif position_type == 'derecha_tension':
        personaje_x = a4_width - personaje_width - 80
        personaje_y = (a4_height - personaje_height) // 2
    else:  # centro_triunfo
        personaje_x = (a4_width - personaje_width) // 2
        personaje_y = (a4_height - personaje_height) // 2 + 100

    # ============ 4. EFECTOS CINEMATOGRÁFICOS ============
    resultado = fondo_final.copy().convert('RGBA')

    # Sombra épica del personaje
    sombra = Image.new('RGBA', resultado.size, (0, 0, 0, 0))
    sombra_draw = ImageDraw.Draw(sombra)

    # Sombra más dramática para este estilo
    shadow_offset = 25
    sombra_draw.ellipse([
        personaje_x + shadow_offset,
        personaje_y + personaje_height - 50,
        personaje_x + personaje_width + shadow_offset + 40,
        personaje_y + personaje_height + 60
    ], fill=(0, 0, 0, 120))

    # Desenfocar sombra
    from PIL import ImageFilter
    sombra = sombra.filter(ImageFilter.GaussianBlur(radius=15))
    resultado = Image.alpha_composite(resultado, sombra)

    # ============ 5. PEGAR PERSONAJE ============
    if personaje_resized.mode == 'RGBA':
        resultado.paste(personaje_resized, (personaje_x, personaje_y), personaje_resized)
    else:
        resultado.paste(personaje_resized, (personaje_x, personaje_y))

    # ============ 6. EFECTOS FINALES CINEMATOGRÁFICOS ============
    # Viñeta más pronunciada
    viñeta = Image.new('RGBA', resultado.size, (0, 0, 0, 0))
    viñeta_draw = ImageDraw.Draw(viñeta)

    center_x, center_y = a4_width // 2, a4_height // 2
    max_radius = max(a4_width, a4_height) // 2

    for i in range(0, max_radius, 15):
        alpha = int(50 * (i / max_radius))  # Viñeta más intensa
        viñeta_draw.ellipse([
            center_x - max_radius + i, center_y - max_radius + i,
            center_x + max_radius - i, center_y + max_radius - i
        ], outline=(0, 0, 0, alpha), width=20)

    resultado = Image.alpha_composite(resultado, viñeta)

    # Realce de color más dramático
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Color(resultado)
    resultado = enhancer.enhance(1.2)  # +20% saturación

    enhancer = ImageEnhance.Contrast(resultado)
    resultado = enhancer.enhance(1.1)  # +10% contraste

    logger.info(f"🎬 Fondo completo épico creado - Página {numero_pagina}: Posición '{position_type}', Escala {scale}")

    return resultado.convert('RGB')

def draw_texto_con_sombra_blanca(draw, x, y, texto, font, color_texto='#2C3E50', max_width=None):
    """📝 Dibuja texto con sombra blanca para máxima legibilidad sobre fondos complejos."""

    # Sombra blanca múltiple para máximo contraste
    sombra_offsets = [
        (-3, -3), (-3, 0), (-3, 3),
        (0, -3), (0, 3),
        (3, -3), (3, 0), (3, 3),
        # Sombra adicional para más contraste
        (-2, -2), (-2, 2), (2, -2), (2, 2)
    ]

    # Dibujar sombras blancas
    for offset_x, offset_y in sombra_offsets:
        draw.text((x + offset_x, y + offset_y), texto, font=font, fill='white')

    # Dibujar texto principal
    if max_width:
        return draw_formatted_line(draw, x, y, texto, {'normal': font}, color_texto, max_width)
    else:
        draw.text((x, y), texto, font=font, fill=color_texto)
        try:
            return draw.textlength(texto, font=font)
        except AttributeError:
            bbox = draw.textbbox((0, 0), texto, font=font)
            return bbox[2] - bbox[0]

# ============================================================================
# FUNCIONES PARA MULTIPÁGINA
# ============================================================================

def calcular_lineas_por_pagina(header_height: int, line_spacing: int = 80) -> int:
    """Calcula cuántas líneas de texto caben en una página."""
    a4_height = 3508
    max_height = 3380
    
    y_text_start = header_height + 245
    espacio_disponible = max_height - y_text_start
    lineas_aproximadas = int(espacio_disponible / line_spacing)
    
    return lineas_aproximadas

def dividir_texto_en_paginas(texto_completo: str, fonts, max_width_px: int, 
                              lineas_por_pagina: int, draw) -> List[List[Tuple[str, str]]]:
    """Divide el texto completo en chunks que quepan en cada página."""
    todas_las_lineas = wrap_text_with_markdown(texto_completo, fonts, max_width_px, draw)
    
    paginas = []
    pagina_actual = []
    lineas_en_pagina_actual = 0
    lineas_primera_pagina = lineas_por_pagina - 3
    
    for linea, tipo in todas_las_lineas:
        max_lineas = lineas_primera_pagina if len(paginas) == 0 else lineas_por_pagina
        
        if tipo == 'paragraph_break':
            pagina_actual.append((linea, tipo))
            lineas_en_pagina_actual += 1
        else:
            if lineas_en_pagina_actual >= max_lineas:
                paginas.append(pagina_actual)
                pagina_actual = []
                lineas_en_pagina_actual = 0
            
            pagina_actual.append((linea, tipo))
            lineas_en_pagina_actual += 1
    
    if pagina_actual:
        paginas.append(pagina_actual)
    
    logger.info(f"📄 Texto dividido en {len(paginas)} páginas")
    return paginas

def crear_pagina_cuento(
    header_img: Image.Image,
    texto_pagina: List[Tuple[str, str]],
    titulo: str,
    es_primera_pagina: bool,
    numero_pagina: int,
    total_paginas: int,
    header_height: int = 1150,
    estilo: str = "infantil"
) -> Image.Image:
    """Crea UNA página del cuento con el diseño completo."""
    logger.info(f"📄 Creando página {numero_pagina}/{total_paginas}")
    
    a4_width = 2480
    a4_height = 3508
    
    canvas = Image.new('RGBA', (a4_width, a4_height), '#FFFEF0' if estilo == "infantil" else 'white')
    
    # PROCESAMIENTO DE IMAGEN
    target_aspect = a4_width / header_height
    image_aspect = header_img.width / header_img.height

    if image_aspect < target_aspect:
        new_width = a4_width
        new_height = int(a4_width / image_aspect)
        header_img_resized = header_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        top_crop = max(0, (new_height - header_height) // 2)
        bottom_crop = top_crop + header_height
        header_img_final = header_img_resized.crop((0, top_crop, new_width, bottom_crop))
    else:
        new_height = header_height
        new_width = int(header_height * image_aspect)
        header_img_resized = header_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left_crop = max(0, (new_width - a4_width) // 2)
        right_crop = left_crop + a4_width
        header_img_final = header_img_resized.crop((left_crop, 0, right_crop, new_height))
    
    canvas.paste(header_img_final, (0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # CARGAR FUENTES
    try:
        font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 52)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
        font_titulo = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 100)
        font_drop_cap_base = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 150)
        font_page_number = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception as e:
        logger.error(f"❌ Error fuentes: {e}")
        font_normal = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_titulo = ImageFont.load_default()
        font_drop_cap_base = ImageFont.load_default()
        font_page_number = ImageFont.load_default()
    
    fonts = {
        'normal': font_normal,
        'bold': font_bold,
        'italic': font_bold,
        'bold_italic': font_bold
    }
    
    # LAYOUT
    margin_left = 160
    margin_right = 160
    line_spacing = 80
    paragraph_spacing = 40
    max_width_px = a4_width - margin_left - margin_right
    max_height = 3380
    
    y_text = header_height + 245
    
    # TÍTULO (solo en primera página)
    if es_primera_pagina and titulo:
        titulo_capitalizado = to_title_case(titulo)
        
        bbox_title = draw.textbbox((0, 0), titulo_capitalizado, font=font_titulo)
        title_width = bbox_title[2] - bbox_title[0]
        title_height = bbox_title[3] - bbox_title[1]
        
        padding_x = 40
        padding_y = 30
        rect_height = title_height + 2 * padding_y
        
        title_x_bg = (a4_width - title_width - 2 * padding_x) // 2
        title_y_bg = header_height - rect_height // 2
        
        title_bg_rect = [
            (title_x_bg, title_y_bg),
            (title_x_bg + title_width + 2 * padding_x, title_y_bg + rect_height)
        ]
        
        title_offset_x = title_x_bg + padding_x
        title_offset_y = title_y_bg + padding_y
        
        alpha_img = Image.new('RGBA', canvas.size, (255, 255, 255, 0))
        alpha_draw = ImageDraw.Draw(alpha_img)
        alpha_draw.rectangle(title_bg_rect, fill=(255, 255, 255, 180))
        canvas = Image.alpha_composite(canvas, alpha_img)
        draw = ImageDraw.Draw(canvas)
        
        title_main_color = '#E91E63'
        title_outline_color = '#8E24AA'
        outline_width = 4
        
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx * dx + dy * dy >= outline_width * outline_width:
                    draw.text((title_offset_x + dx, title_offset_y + dy), titulo_capitalizado, 
                             font=font_titulo, fill=title_outline_color)
        
        draw.text((title_offset_x, title_offset_y), titulo_capitalizado, font=font_titulo, fill=title_main_color)
    
    canvas = canvas.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    
    text_color = '#2C3E50' if estilo == "infantil" else '#2c2c2c'
    
    # DIBUJAR TEXTO
    if es_primera_pagina and texto_pagina and texto_pagina[0][1] == 'text':
        full_first_line_content = texto_pagina[0][0]
        drop_cap_char = full_first_line_content[0]
        
        DROP_CAP_LINES = 3
        drop_cap_size = line_spacing * (DROP_CAP_LINES + 0.3)
        
        try:
            font_drop_cap = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 
                                               int(drop_cap_size))
        except Exception:
            font_drop_cap = font_drop_cap_base
        
        bbox_cap = draw.textbbox((0, 0), drop_cap_char, font=font_drop_cap)
        cap_width = bbox_cap[2] - bbox_cap[0]
        
        cap_y_adjustment = -160
        drop_cap_x = margin_left
        drop_cap_y_final = y_text + cap_y_adjustment
        
        cap_color = '#ef4444'
        draw.text((drop_cap_x, drop_cap_y_final), drop_cap_char, font=font_drop_cap, fill=cap_color)
        
        rest_x = drop_cap_x + cap_width + 25
        rest_max_width = a4_width - rest_x - margin_right
        
        first_line_without_cap = full_first_line_content[1:].lstrip()
        
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        wrapped_first_para = wrap_text_with_markdown(first_line_without_cap, fonts, rest_max_width, temp_draw)
        
        y_current = y_text
        lines_beside_cap = 0
        
        for line_content, _ in wrapped_first_para:
            if lines_beside_cap < DROP_CAP_LINES:
                if line_content.strip():
                    draw_formatted_line(draw, rest_x, y_current, line_content, fonts, text_color, 
                                       max_width_px=rest_max_width)
                y_current += line_spacing
                lines_beside_cap += 1
            else:
                break
        
        y_text = y_current
        
        for i in range(lines_beside_cap, len(wrapped_first_para)):
            line_content, _ = wrapped_first_para[i]
            if y_text > max_height:
                break
            draw_formatted_line(draw, margin_left, y_text, line_content, fonts, text_color, 
                               max_width_px=max_width_px)
            y_text += line_spacing
        
        for line, line_type in texto_pagina[1:]:
            if y_text > max_height:
                break
            
            if line_type == 'paragraph_break':
                y_text += paragraph_spacing
                continue
            
            draw_formatted_line(draw, margin_left, y_text, line, fonts, text_color, 
                               max_width_px=max_width_px)
            y_text += line_spacing
    else:
        for line, line_type in texto_pagina:
            if y_text > max_height:
                break
            
            if line_type == 'paragraph_break':
                y_text += paragraph_spacing
                continue
            
            draw_formatted_line(draw, margin_left, y_text, line, fonts, text_color, 
                               max_width_px=max_width_px)
            y_text += line_spacing
    
    # NÚMERO DE PÁGINA
    if total_paginas > 1:
        page_text = f"{numero_pagina}"
        bbox_page = draw.textbbox((0, 0), page_text, font=font_page_number)
        page_width = bbox_page[2] - bbox_page[0]
        page_x = a4_width - margin_right - page_width
        page_y = a4_height - 150
        
        draw.text((page_x, page_y), page_text, font=font_page_number, fill='#999999')
    
    if estilo == "infantil":
        draw_wavy_border(draw, a4_width, a4_height)
    
    return canvas

def imagenes_a_pdf(imagenes: List[Image.Image], output_path: str):
    """Convierte una lista de imágenes PIL a un PDF multipágina."""
    if not imagenes:
        raise ValueError("No hay imágenes para convertir a PDF")
    
    imagenes_rgb = [img.convert('RGB') if img.mode != 'RGB' else img for img in imagenes]
    
    imagenes_rgb[0].save(
        output_path,
        save_all=True,
        append_images=imagenes_rgb[1:],
        resolution=300.0,
        quality=95
    )
    
    logger.info(f"✅ PDF creado con {len(imagenes)} páginas: {output_path}")

# ============================================================================
# 🆕 ENDPOINT: COMBINAR DOCUMENTOS
# ============================================================================

@app.post("/combinar-documentos")
async def combinar_documentos(request: CombinarDocumentosRequest):
    """
    Combina múltiples imágenes PNG o PDFs en un solo PDF multipágina.
    
    Body JSON:
    {
        "rutas_archivos": ["/tmp/file1.png", "/tmp/file2.png", "/tmp/file3.png"]
    }
    """
    logger.info(f"🔗 COMBINAR DOCUMENTOS: {len(request.rutas_archivos)} archivos")
    
    try:
        if not request.rutas_archivos:
            raise HTTPException(status_code=400, detail="No se proporcionaron archivos")
        
        imagenes_combinadas = []
        
        for i, ruta in enumerate(request.rutas_archivos):
            logger.info(f"📄 Procesando {i+1}/{len(request.rutas_archivos)}: {ruta}")
            
            if not os.path.exists(ruta):
                logger.warning(f"⚠️ Archivo no encontrado: {ruta}")
                continue
            
            extension = os.path.splitext(ruta)[1].lower()
            
            if extension in ['.png', '.jpg', '.jpeg']:
                img = Image.open(ruta)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                imagenes_combinadas.append(img)
                
            elif extension == '.pdf':
                try:
                    from pdf2image import convert_from_path
                    paginas_pdf = convert_from_path(ruta, dpi=300)
                    imagenes_combinadas.extend(paginas_pdf)
                except ImportError:
                    logger.error("❌ pdf2image no instalado")
                    raise HTTPException(status_code=500, detail="pdf2image no disponible")
            else:
                logger.warning(f"⚠️ Formato no soportado: {extension}")
        
        if not imagenes_combinadas:
            raise HTTPException(status_code=400, detail="No hay imágenes válidas")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Cuento_Completo_{len(imagenes_combinadas)}pag_{timestamp}.pdf"
        output_path = f"/tmp/{filename}"
        
        imagenes_a_pdf(imagenes_combinadas, output_path)
        
        logger.info(f"✅ PDF combinado: {len(imagenes_combinadas)} páginas")
        
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=filename,
            headers={
                "X-Total-Pages": str(len(imagenes_combinadas)),
                "X-Files-Combined": str(len(request.rutas_archivos))
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENDPOINT: CREAR CUENTO MULTIPÁGINA
# ============================================================================

@app.post("/crear-cuento-multipagina")
async def crear_cuento_multipagina(
    imagen: UploadFile = File(...),
    texto_cuento: str = Form(...),
    titulo: str = Form(default=""),
    header_height: int = Form(default=1150),
    estilo: str = Form(default="infantil"),
):
    """Crea un cuento de múltiples páginas automáticamente."""
    logger.info(f"📚 CUENTO MULTIPÁGINA: {len(texto_cuento)} caracteres")
    
    try:
        img_bytes = await imagen.read()
        header_img = Image.open(io.BytesIO(img_bytes))
        
        if header_img.mode != 'RGB':
            header_img = header_img.convert('RGB')
        
        margin_left = 160
        margin_right = 160
        max_width_px = 2480 - margin_left - margin_right
        line_spacing = 80
        
        lineas_por_pagina = calcular_lineas_por_pagina(header_height, line_spacing)
        logger.info(f"📐 ~{lineas_por_pagina} líneas por página")
        
        try:
            font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 52)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
        except:
            font_normal = ImageFont.load_default()
            font_bold = ImageFont.load_default()
        
        fonts = {
            'normal': font_normal,
            'bold': font_bold,
            'italic': font_bold,
            'bold_italic': font_bold
        }
        
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        paginas_texto = dividir_texto_en_paginas(
            texto_cuento, 
            fonts, 
            max_width_px, 
            lineas_por_pagina, 
            temp_draw
        )
        
        total_paginas = len(paginas_texto)
        logger.info(f"📄 {total_paginas} páginas")
        
        imagenes_paginas = []
        
        for i, texto_pagina in enumerate(paginas_texto):
            es_primera = (i == 0)
            numero_pagina = i + 1
            
            pagina_img = crear_pagina_cuento(
                header_img=header_img,
                texto_pagina=texto_pagina,
                titulo=titulo,
                es_primera_pagina=es_primera,
                numero_pagina=numero_pagina,
                total_paginas=total_paginas,
                header_height=header_height,
                estilo=estilo
            )
            
            imagenes_paginas.append(pagina_img)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        titulo_sanitizado = sanitize_filename(titulo) if titulo else "Sin_Titulo"
        filename = f"Cuento_{titulo_sanitizado}_{total_paginas}pag_{timestamp}.pdf"
        
        output_path = f"/tmp/{filename}"
        imagenes_a_pdf(imagenes_paginas, output_path)
        
        palabras_aprox = len(texto_cuento.split())
        logger.info(f"✅ Cuento: {total_paginas} pág, ~{palabras_aprox} palabras")
        
        return FileResponse(
            output_path, 
            media_type="application/pdf", 
            filename=filename,
            headers={
                "X-Total-Pages": str(total_paginas),
                "X-Approx-Words": str(palabras_aprox)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENDPOINT: CREAR FICHA (1 PÁGINA)
# ============================================================================

@app.post("/crear-ficha")
async def crear_ficha(
    imagen_fondo: UploadFile = File(...),
    imagen_personaje: UploadFile = File(...),
    texto_cuento: str = Form(...),
    titulo: str = Form(default=""),
    header_height: int = Form(default=1150),
    estilo: str = Form(default="infantil"),
    tipo_composicion: str = Form(default="header_texto"),
    es_primera_pagina: bool = Form(default=True),
    numero_pagina: int = Form(default=1),
    total_paginas: int = Form(default=999),
):
    """Crea UNA sola página (compatibilidad)."""
    logger.info(f"📥 FICHA SIMPLE: {len(texto_cuento)} chars")
    
    palabras = len(texto_cuento.split())
    if palabras > 270:
        logger.warning(f"⚠️ Texto largo ({palabras} palabras)")

    try:
        # Procesar imagen fondo
        fondo_bytes = await imagen_fondo.read()
        fondo_img = Image.open(io.BytesIO(fondo_bytes))
        if fondo_img.mode != 'RGB':
            fondo_img = fondo_img.convert('RGB')

        # Procesar imagen personaje y remover fondo blanco
        personaje_bytes = await imagen_personaje.read()
        personaje_img = Image.open(io.BytesIO(personaje_bytes))
        personaje_img = remover_fondo_blanco(personaje_img)

        # ============ ELEGIR TIPO DE COMPOSICIÓN ============
        a4_width = 2480
        a4_height = 3508

        if tipo_composicion == "fondo_completo":
            # 🎬 ESTILO FONDO COMPLETO - Página completa épica
            logger.info(f"🎬 Creando estilo FONDO COMPLETO épico")

            # Crear fondo completo con personaje grande (YA ES LA PÁGINA COMPLETA)
            canvas = crear_fondo_completo_epico(fondo_img, personaje_img, a4_width, a4_height, numero_pagina)
            draw = ImageDraw.Draw(canvas)

            # Cargar fuentes
            try:
                font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 58)  # Más grande
                font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 58)
                font_titulo = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 120)
            except:
                font_normal = ImageFont.load_default()
                font_bold = ImageFont.load_default()
                font_titulo = ImageFont.load_default()

            # TÍTULO con sombra blanca épica (solo primera página)
            if es_primera_pagina and titulo:
                titulo_capitalizado = to_title_case(titulo)
                bbox_title = draw.textbbox((0, 0), titulo_capitalizado, font=font_titulo)
                title_width = bbox_title[2] - bbox_title[0]

                title_x = (a4_width - title_width) // 2
                title_y = 150  # Arriba de la página

                draw_texto_con_sombra_blanca(draw, title_x, title_y, titulo_capitalizado, font_titulo, '#FFD700')

            # TEXTO con sombra blanca - buscar mejor zona libre
            line_spacing = 75
            margin_left = 150
            margin_right = 150
            max_width_texto = a4_width - margin_left - margin_right

            # Detectar zona más libre para el texto
            zonas_texto = [
                {'y_start': 400, 'y_end': 1200, 'nombre': 'superior'},      # Zona superior
                {'y_start': 1200, 'y_end': 2200, 'nombre': 'media'},        # Zona media
                {'y_start': 2200, 'y_end': 3200, 'nombre': 'inferior'}      # Zona inferior
            ]

            # Usar zona media como default (generalmente más libre)
            zona_elegida = zonas_texto[1]
            texto_y_start = zona_elegida['y_start'] + 100
            texto_y_end = zona_elegida['y_end'] - 100

            logger.info(f"📝 Zona de texto: {zona_elegida['nombre']} ({texto_y_start}-{texto_y_end})")

            # Dividir texto en párrafos y procesar línea por línea
            paragrafos = texto_cuento.strip().split('\n\n')
            current_y = texto_y_start
            max_lines = int((texto_y_end - texto_y_start) / line_spacing)

            lines_used = 0
            for i, parrafo in enumerate(paragrafos):
                if lines_used >= max_lines - 1:
                    break

                # Dividir párrafo en líneas que quepan en el ancho
                palabras = parrafo.split()
                linea_actual = []

                for palabra in palabras:
                    test_line = ' '.join(linea_actual + [palabra])
                    try:
                        test_width = draw.textlength(test_line, font=font_normal)
                    except AttributeError:
                        bbox = draw.textbbox((0, 0), test_line, font=font_normal)
                        test_width = bbox[2] - bbox[0]

                    if test_width <= max_width_texto - 100:  # Margen extra para sombras
                        linea_actual.append(palabra)
                    else:
                        # Dibujar línea actual
                        if linea_actual and current_y <= texto_y_end:
                            linea_text = ' '.join(linea_actual)
                            draw_texto_con_sombra_blanca(draw, margin_left, current_y, linea_text, font_normal)
                            current_y += line_spacing
                            lines_used += 1

                        # Empezar nueva línea
                        linea_actual = [palabra]

                # Dibujar línea final del párrafo
                if linea_actual and current_y <= texto_y_end and lines_used < max_lines:
                    linea_text = ' '.join(linea_actual)
                    draw_texto_con_sombra_blanca(draw, margin_left, current_y, linea_text, font_normal)
                    current_y += line_spacing
                    lines_used += 1

                # Espacio entre párrafos
                if i < len(paragrafos) - 1 and lines_used < max_lines - 1:
                    current_y += line_spacing * 0.5
                    lines_used += 0.5

            logger.info(f"📝 Texto renderizado: {lines_used:.1f}/{max_lines} líneas")

            # Número de página
            if total_paginas > 1:
                page_text = f"{numero_pagina}"
                draw_texto_con_sombra_blanca(draw, a4_width - 200, a4_height - 150, page_text, font_bold, '#FFD700')

            pagina_img = canvas

        else:
            # 🎨 ESTILO HEADER+TEXTO ORIGINAL (por defecto)
            logger.info(f"🎨 Creando estilo HEADER+TEXTO original")

            # Combinar fondo + personaje con efectos épicos
            header_img = combinar_fondo_personaje(fondo_img, personaje_img, header_height, numero_pagina, total_paginas)

            margin_left = 160
            margin_right = 160
            max_width_px = 2480 - margin_left - margin_right

            try:
                font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 52)
                font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
            except:
                font_normal = ImageFont.load_default()
                font_bold = ImageFont.load_default()

            fonts = {
                'normal': font_normal,
                'bold': font_bold,
                'italic': font_bold,
                'bold_italic': font_bold
            }

            temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            texto_lines = wrap_text_with_markdown(texto_cuento, fonts, max_width_px, temp_draw)

            pagina_img = crear_pagina_cuento(
                header_img=header_img,
                texto_pagina=texto_lines,
                titulo=titulo,
                es_primera_pagina=es_primera_pagina,
                numero_pagina=numero_pagina,
                total_paginas=total_paginas,
                header_height=header_height,
                estilo=estilo
            )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        titulo_sanitizado = sanitize_filename(titulo) if titulo else "Sin_Titulo"
        filename = f"Cuento_{titulo_sanitizado}_ficha_{timestamp}.png"
        
        output_path = f"/tmp/{filename}"
        pagina_img.save(output_path, quality=95, dpi=(300, 300))
        
        logger.info(f"✅ Ficha creada: {filename}")
        
        return FileResponse(
            output_path,
            media_type="image/png",
            filename=filename,
            headers={
                "rutas_archivos": output_path
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENDPOINTS DE INFO
# ============================================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "9.0-COMBINAR",
        "features": ["crear_cuento_multipagina", "crear_ficha", "combinar_documentos"],
        "endpoints": {
            "POST /crear-cuento-multipagina": "Crea cuentos multipágina automático (PDF)",
            "POST /crear-ficha": "Crea ficha de 1 página (PNG)",
            "POST /combinar-documentos": "🆕 Combina múltiples PNGs/PDFs en un PDF"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "version": "9.0-COMBINAR"}
