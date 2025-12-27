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
    portada: str = None  # Base64 de la imagen de portada (opcional)
    titulo: str = None   # Título del cuento para la portada

class FichaCuadradaRequest(BaseModel):
    texto: str
    tamano: int = 1200  # Tamaño de la ficha cuadrada en píxeles
    color_fondo: str = "#FFFFFF"  # Color de fondo en hexadecimal
    color_texto: str = "#2c2c2c"  # Color del texto en hexadecimal

class CombinarHojasCuadradasRequest(BaseModel):
    rutas_images: List[str]  # ["/tmp/img1.png", "/tmp/img2.png", ...]
    rutas_textos: List[str]  # ["/tmp/txt1.png", "/tmp/txt2.png", ...]
    portada: str = None  # Base64 de la imagen de portada (opcional)
    titulo: str = None   # Título del cuento para la portada

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
        else:  # inferior_der o centrom
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

def crear_fondo_solo_imagen(fondo_img: Image.Image, a4_width: int, a4_height: int) -> Image.Image:
    """🖼️ FONDO SIMPLE - Solo ajusta la imagen completa al tamaño A4 sin agregar personajes."""

    logger.info(f"🖼️ Creando fondo simple: {a4_width}x{a4_height}")

    # ============ 1. FONDO AJUSTADO A A4 COMPLETO ============
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

    logger.info(f"🖼️ Fondo simple creado correctamente")
    return fondo_final.convert('RGB')

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

def crear_burbuja_texto(draw, x, y, texto, font, color_texto='white'):
    """💬 Crea una burbuja semitransparente elegante para texto legible."""

    # Calcular dimensiones del texto
    try:
        bbox = draw.textbbox((0, 0), texto, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width = len(texto) * 15
        text_height = 30

    # Configurar burbuja
    padding = 15  # Espacio interno alrededor del texto
    border_radius = 12  # Radio para esquinas redondeadas

    # Dimensiones de la burbuja
    bubble_x = x - padding
    bubble_y = y - padding
    bubble_width = text_width + (padding * 2)
    bubble_height = text_height + (padding * 2)

    # ============ CREAR BURBUJA SEMITRANSPARENTE ============
    # Color de fondo semitransparente (negro con 70% opacidad)
    bubble_color = (0, 0, 0, 180)  # RGBA: Negro con alpha 180/255

    # Crear imagen temporal para la burbuja
    bubble_img = Image.new('RGBA', (int(bubble_width), int(bubble_height)), (0, 0, 0, 0))
    bubble_draw = ImageDraw.Draw(bubble_img)

    # Dibujar rectángulo redondeado
    bubble_draw.rounded_rectangle(
        [0, 0, bubble_width, bubble_height],
        radius=border_radius,
        fill=bubble_color
    )

    # Obtener la imagen base para pegar la burbuja
    try:
        canvas_img = draw._image if hasattr(draw, '_image') else None
        if canvas_img:
            canvas_img.paste(bubble_img, (int(bubble_x), int(bubble_y)), bubble_img)
    except:
        # Fallback: dibujar rectángulo simple si hay problemas
        draw.rectangle([bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height],
                      fill=(0, 0, 0, 180))

    # ============ DIBUJAR TEXTO BLANCO LIMPIO ============
    draw.text((x, y), texto, font=font, fill=color_texto)

    return text_width

def draw_texto_con_sombra_blanca(draw, x, y, texto, font, color_texto='white', max_width=None):
    """💬 Dibuja texto en burbuja semitransparente elegante."""
    return crear_burbuja_texto(draw, x, y, texto, font, color_texto)

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

def crear_portada_cuadrada_desde_base64(portada_base64: str, titulo: str = "Mi Cuento") -> Image.Image:
    """Crea una portada CUADRADA hermosa desde imagen base64 SIN cortar la imagen original."""
    logger.info(f"🔍 DEBUG en crear_portada_cuadrada_desde_base64: Título recibido: '{titulo}'")
    import base64

    # Decodificar base64
    portada_bytes = base64.b64decode(portada_base64)
    portada_img = Image.open(io.BytesIO(portada_bytes))
    if portada_img.mode != 'RGB':
        portada_img = portada_img.convert('RGB')

    logger.info(f"📐 DEBUG: Imagen original desde base64: {portada_img.width}x{portada_img.height}")

    # ============ MANTENER IMAGEN ORIGINAL + AGREGAR TÍTULO ============
    # NO redimensionar, NO cortar, solo agregar título

    canvas = portada_img.copy()  # Usar imagen original tal como es
    draw = ImageDraw.Draw(canvas)

    # ============ CARGAR FUENTES PROPORCIONALES ============
    imagen_size = min(portada_img.width, portada_img.height)
    font_size = max(60, imagen_size // 20)  # Proporcional al tamaño de imagen

    try:
        font_titulo_grande = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", font_size)
        logger.info(f"✅ Fuente cargada: tamaño {font_size}px para imagen {imagen_size}px")
    except:
        font_titulo_grande = ImageFont.load_default()

    # ============ TÍTULO DORADO EN LA PARTE INFERIOR ============
    logger.info(f"🔍 DEBUG: Evaluando título - titulo: '{titulo}', titulo.strip(): '{titulo.strip() if titulo else None}'")
    if titulo and titulo.strip():
        logger.info(f"✅ DEBUG: Título válido, procediendo a renderizar...")
        titulo_capitalizado = to_title_case(titulo)
        logger.info(f"🔍 DEBUG: Título capitalizado: '{titulo_capitalizado}'")

        # Configurar área de texto
        margin = imagen_size * 0.05  # 5% margen
        max_width = portada_img.width - (2 * margin)

        # Dividir título en líneas
        palabras = titulo_capitalizado.split()
        lineas_titulo = []
        linea_actual = []

        for palabra in palabras:
            test_line = ' '.join(linea_actual + [palabra])
            try:
                ancho_test = draw.textlength(test_line, font=font_titulo_grande)
            except AttributeError:
                bbox = draw.textbbox((0, 0), test_line, font=font_titulo_grande)
                ancho_test = bbox[2] - bbox[0]

            if ancho_test <= max_width:
                linea_actual.append(palabra)
            else:
                if linea_actual:
                    lineas_titulo.append(' '.join(linea_actual))
                linea_actual = [palabra]

        if linea_actual:
            lineas_titulo.append(' '.join(linea_actual))

        # Dibujar título en la parte inferior
        if lineas_titulo:
            try:
                altura_linea = draw.textbbox((0, 0), "Ag", font=font_titulo_grande)[3] - draw.textbbox((0, 0), "Ag", font=font_titulo_grande)[1]
            except:
                altura_linea = font_size

            interlineado = altura_linea * 0.3
            altura_total = len(lineas_titulo) * altura_linea + (len(lineas_titulo) - 1) * interlineado

            # Posicionar en parte inferior
            y_start = portada_img.height - altura_total - margin

            color_dorado = "#FFD700"
            color_sombra = "#8B4513"

            y_actual = y_start
            for linea in lineas_titulo:
                try:
                    ancho_linea = draw.textlength(linea, font=font_titulo_grande)
                except AttributeError:
                    bbox = draw.textbbox((0, 0), linea, font=font_titulo_grande)
                    ancho_linea = bbox[2] - bbox[0]

                x_centrado = (portada_img.width - ancho_linea) // 2

                # Sombra
                draw.text((x_centrado + 3, y_actual + 3), linea, font=font_titulo_grande, fill=color_sombra)
                # Texto principal
                draw.text((x_centrado, y_actual), linea, font=font_titulo_grande, fill=color_dorado)

                y_actual += altura_linea + interlineado

        logger.info(f"✅ Portada cuadrada desde base64 creada con título")
    else:
        logger.info(f"❌ DEBUG: Título NO válido - titulo: '{titulo}', es None: {titulo is None}, es vacío: {not titulo.strip() if titulo else True}")
        logger.info(f"📖 Portada creada desde base64 SIN título adicional")

    return canvas.convert('RGB')

def crear_portada_desde_base64(portada_base64: str, titulo: str = "Mi Cuento") -> Image.Image:
    """Crea una portada hermosa desde imagen base64."""
    logger.info(f"🔍 DEBUG en crear_portada_desde_base64: Título recibido: '{titulo}'")
    import base64

    # Decodificar base64
    portada_bytes = base64.b64decode(portada_base64)
    portada_img = Image.open(io.BytesIO(portada_bytes))
    if portada_img.mode != 'RGB':
        portada_img = portada_img.convert('RGB')

    # Dimensiones A4
    a4_width = 2480
    a4_height = 3508

    # ============ CREAR FONDO DE PORTADA A4 COMPLETO ============
    portada_aspect = portada_img.width / portada_img.height
    page_aspect = a4_width / a4_height

    if portada_aspect < page_aspect:
        new_width = a4_width
        new_height = int(a4_width / portada_aspect)
        portada_resized = portada_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        y_offset = (new_height - a4_height) // 2
        canvas = portada_resized.crop((0, y_offset, a4_width, y_offset + a4_height))
    else:
        new_height = a4_height
        new_width = int(a4_height * portada_aspect)
        portada_resized = portada_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        x_offset = (new_width - a4_width) // 2
        canvas = portada_resized.crop((x_offset, 0, x_offset + a4_width, a4_height))

    if canvas.size != (a4_width, a4_height):
        canvas = canvas.resize((a4_width, a4_height), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(canvas)

    # ============ CARGAR FUENTES ELEGANTES ============
    try:
        font_titulo_grande = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 140)
    except:
        font_titulo_grande = ImageFont.load_default()

    # ============ TÍTULO DORADO ESPECTACULAR EN LA PARTE INFERIOR ============
    logger.info(f"🔍 DEBUG: Evaluando título - titulo: '{titulo}', titulo.strip(): '{titulo.strip() if titulo else None}'")
    if titulo and titulo.strip():
        logger.info(f"✅ DEBUG: Título válido, procediendo a renderizar...")
        titulo_capitalizado = to_title_case(titulo)
        logger.info(f"🔍 DEBUG: Título capitalizado: '{titulo_capitalizado}'")

        # ============ FUENTE MÁS GRANDE Y ELEGANTE ============
        try:
            font_titulo_epico = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 180)
        except:
            font_titulo_epico = font_titulo_grande

        bbox_titulo = draw.textbbox((0, 0), titulo_capitalizado, font=font_titulo_epico)
        titulo_width = bbox_titulo[2] - bbox_titulo[0]
        titulo_height = bbox_titulo[3] - bbox_titulo[1]

        # ============ POSICIÓN: 25% DESDE ABAJO ============
        titulo_x = (a4_width - titulo_width) // 2
        titulo_y = a4_height - (a4_height * 0.25) - titulo_height // 2  # 25% desde abajo

        # ============ BURBUJA ELEGANTE MÁS GRANDE ============
        padding_titulo = 80  # Padding más generoso
        bubble_titulo_width = titulo_width + (padding_titulo * 2)
        bubble_titulo_height = titulo_height + (padding_titulo * 2)
        bubble_titulo_x = titulo_x - padding_titulo
        bubble_titulo_y = titulo_y - padding_titulo

        # ============ FONDO NEGRO SEMITRANSPARENTE ELEGANTE ============
        bubble_titulo_img = Image.new('RGBA', (int(bubble_titulo_width), int(bubble_titulo_height)), (0, 0, 0, 0))
        bubble_titulo_draw = ImageDraw.Draw(bubble_titulo_img)

        bubble_titulo_draw.rounded_rectangle(
            [0, 0, bubble_titulo_width, bubble_titulo_height],
            radius=40,  # Radio más grande
            fill=(0, 0, 0, 160),  # Fondo negro elegante con transparencia
            outline=(50, 50, 50, 255),  # Borde más oscuro
            width=4  # Borde más grueso
        )

        canvas.paste(bubble_titulo_img, (int(bubble_titulo_x), int(bubble_titulo_y)), bubble_titulo_img)

        # ============ TEXTO DORADO CON EFECTOS ESPECTACULARES ============
        # Sombra dorada profunda (múltiples capas)
        shadow_offsets = [(-6, -6), (-4, -4), (-2, -2), (6, 6), (4, 4), (2, 2)]
        shadow_color = '#B8860B'  # Oro oscuro para sombra

        for dx, dy in shadow_offsets:
            draw.text((titulo_x + dx, titulo_y + dy), titulo_capitalizado, font=font_titulo_epico, fill=shadow_color)

        # ============ TEXTO PRINCIPAL DORADO BRILLANTE ============
        # Degradado dorado simulado con múltiples tonos
        colores_dorados = [
            '#FFD700',  # Oro brillante principal
            '#FFA500',  # Naranja dorado
            '#FFFF00',  # Amarillo brillante
        ]

        # Aplicar múltiples capas de colores dorados
        for i, color in enumerate(colores_dorados):
            offset = i * 1
            draw.text((titulo_x + offset, titulo_y + offset), titulo_capitalizado, font=font_titulo_epico, fill=color)

        # ============ BRILLO FINAL DORADO ============
        draw.text((titulo_x, titulo_y), titulo_capitalizado, font=font_titulo_epico, fill='#FFFF99')  # Brillo final

        logger.info(f"✨ Portada con título DORADO espectacular: '{titulo[:30]}...'")
    else:
        logger.info(f"❌ DEBUG: Título NO válido - titulo: '{titulo}', es None: {titulo is None}, es vacío: {not titulo if titulo is not None else 'N/A'}")
        logger.info(f"📖 Portada creada desde base64 SIN título adicional")

    return canvas.convert('RGB')

def crear_ficha_cuadrada_texto(texto: str, tamano: int = 1200, color_fondo: str = "#FFFFFF", color_texto: str = "#2c2c2c", altura: int = None) -> Image.Image:
    """Crea una ficha cuadrada con texto elegante para niños de 10 años."""
    altura_final = altura if altura else tamano
    logger.info(f"📄 Creando ficha de texto {tamano}x{altura_final} con {len(texto)} caracteres")
    logger.info(f"📄 Texto a renderizar: '{texto[:50]}...'")

    # Crear canvas (cuadrado o rectangular si se especifica altura)
    altura_final = altura if altura else tamano
    canvas = Image.new('RGB', (tamano, altura_final), color_fondo)
    draw = ImageDraw.Draw(canvas)

    # Configurar fuentes legibles para niños (tamaño más pequeño)
    try:
        # Fuente más pequeña para que quepa bien
        font_size_base = max(38, tamano // 32)  # Reducido de //28 a //32
        font_texto = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size_base)
        # Cargar fuente en negrilla para preguntas y exclamaciones
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size_base)
        logger.info(f"✅ Fuentes infantiles cargadas: Normal {font_size_base}px, Bold {font_size_base}px")
    except Exception as e:
        logger.warning(f"⚠️ Error cargando fuentes personalizadas: {e}")
        font_texto = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_size_base = 38  # Tamaño más pequeño

    # Configurar márgenes GENEROSOS para cuentos infantiles
    margen_horizontal = tamano * 0.15  # 15% de margen horizontal (más legible)
    margen_vertical = altura_final * 0.20  # 20% de margen vertical (más espacio)
    area_texto_ancho = tamano - (2 * margen_horizontal)
    area_texto_alto = altura_final - (2 * margen_vertical)

    logger.info(f"📖 Márgenes infantiles: H={margen_horizontal:.0f}px ({(margen_horizontal/tamano)*100:.0f}%), V={margen_vertical:.0f}px ({(margen_vertical/altura_final)*100:.0f}%)")

    # PROCESAR TEXTO CON PÁRRAFOS Y FORMATO ESPECIAL
    def procesar_texto_infantil(texto):
        # Dividir en oraciones (salto después de punto)
        import re
        oraciones = re.split(r'(\. )', texto)
        parrafos = []

        for i in range(0, len(oraciones), 2):
            if i < len(oraciones):
                oracion = oraciones[i]
                if i + 1 < len(oraciones):
                    oracion += oraciones[i + 1]  # Agregar el punto
                parrafos.append(oracion.strip())

        return [p for p in parrafos if p]  # Filtrar vacíos

    def detectar_formato_especial(texto):
        # Detectar preguntas y exclamaciones para negrilla
        return bool(re.search(r'[¿?¡!]', texto))

    # Procesar texto en párrafos
    parrafos = procesar_texto_infantil(texto)
    lineas_con_formato = []  # Lista de (texto, es_bold)

    for parrafo in parrafos:
        palabras = parrafo.split()
        linea_actual = []
        es_bold = detectar_formato_especial(parrafo)
        fuente_usar = font_bold if es_bold else font_texto

        for palabra in palabras:
            test_line = ' '.join(linea_actual + [palabra])
            try:
                test_width = draw.textlength(test_line, font=fuente_usar)
            except AttributeError:
                bbox = draw.textbbox((0, 0), test_line, font=fuente_usar)
                test_width = bbox[2] - bbox[0]

            if test_width <= area_texto_ancho:
                linea_actual.append(palabra)
            else:
                if linea_actual:
                    lineas_con_formato.append((' '.join(linea_actual), es_bold))
                linea_actual = [palabra]

        if linea_actual:
            lineas_con_formato.append((' '.join(linea_actual), es_bold))

        # Agregar línea vacía después de cada párrafo (salto de párrafo)
        lineas_con_formato.append(('', False))

    # Remover última línea vacía si existe
    if lineas_con_formato and lineas_con_formato[-1][0] == '':
        lineas_con_formato.pop()

    logger.info(f"📖 Procesamiento infantil: {len(parrafos)} párrafos → {len(lineas_con_formato)} líneas")

    # Calcular espaciado entre líneas con nuevo formato
    if lineas_con_formato:
        try:
            altura_linea = draw.textbbox((0, 0), "Ag", font=font_texto)[3] - draw.textbbox((0, 0), "Ag", font=font_texto)[1]
        except:
            altura_linea = font_size_base

        # Calcular espaciado moderado entre líneas (reducir interlineado)
        interlineado_infantil = altura_linea * 0.25  # 25% interlineado (era 50%, demasiado)
        espacio_total_texto = len(lineas_con_formato) * altura_linea + (len(lineas_con_formato) - 1) * interlineado_infantil

        # Si el texto no cabe, reducir tamaño de fuente
        while espacio_total_texto > area_texto_alto and font_size_base > 30:  # Mínimo 30px
            font_size_base -= 3  # Reducir de a 3px
            try:
                font_texto = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size_base)
                font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size_base)
            except:
                font_texto = ImageFont.load_default()
                font_bold = ImageFont.load_default()

            altura_linea = draw.textbbox((0, 0), "Ag", font=font_texto)[3] - draw.textbbox((0, 0), "Ag", font=font_texto)[1]
            interlineado_infantil = altura_linea * 0.25
            espacio_total_texto = len(lineas_con_formato) * altura_linea + (len(lineas_con_formato) - 1) * interlineado_infantil

        logger.info(f"📖 Tipografía infantil: {font_size_base}px, interlineado: {interlineado_infantil:.1f}px")

    # Centrar texto verticalmente con los nuevos márgenes
    y_inicio = margen_vertical + (area_texto_alto - espacio_total_texto) // 2

    # Dibujar cada línea con formato centrada horizontalmente
    y_actual = y_inicio
    for texto_linea, es_bold in lineas_con_formato:
        if texto_linea:  # Solo dibujar si no es línea vacía
            fuente_usar = font_bold if es_bold else font_texto
            try:
                ancho_linea = draw.textlength(texto_linea, font=fuente_usar)
            except AttributeError:
                bbox = draw.textbbox((0, 0), texto_linea, font=fuente_usar)
                ancho_linea = bbox[2] - bbox[0]

            x_centrado = (tamano - ancho_linea) // 2
            draw.text((x_centrado, y_actual), texto_linea, font=fuente_usar, fill=color_texto)

        y_actual += altura_linea + interlineado_infantil

    # Agregar borde sutil (opcional)
    borde_color = "#e0e0e0"
    draw.rectangle([2, 2, tamano-3, altura_final-3], outline=borde_color, width=2)

    logger.info(f"✅ Ficha cuadrada creada: {len(lineas_con_formato)} líneas, fuente {font_size_base}px")
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
        "rutas_archivos": ["/tmp/file1.png", "/tmp/file2.png", "/tmp/file3.png"],
        "portada": "base64_string_opcional"
    }
    """
    logger.info(f"🔗 COMBINAR DOCUMENTOS: {len(request.rutas_archivos)} archivos + portada: {bool(request.portada)}")
    logger.info(f"🔍 DEBUG: Título recibido: '{request.titulo}'")

    try:
        if not request.rutas_archivos:
            raise HTTPException(status_code=400, detail="No se proporcionaron archivos")

        imagenes_combinadas = []

        # ============ AGREGAR PORTADA PRIMERO SI EXISTE ============
        if request.portada:
            try:
                logger.info("📖 Procesando portada desde base64...")
                # Usar el título del request para la portada
                titulo_para_portada = request.titulo or ""  # Si no hay título, usar string vacío
                portada_img = crear_portada_desde_base64(request.portada, titulo_para_portada)
                imagenes_combinadas.append(portada_img)
                logger.info("✅ Portada agregada como primera página")
            except Exception as e:
                logger.error(f"❌ Error procesando portada: {e}")
                # Continuar sin portada si hay error

        # ============ PROCESAR PÁGINAS DEL CUENTO ============
        
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
        
        total_paginas = len(imagenes_combinadas)
        paginas_cuento = len(request.rutas_archivos)
        tiene_portada = 1 if request.portada else 0

        logger.info(f"✅ PDF combinado: {total_paginas} páginas (portada: {tiene_portada}, cuento: {paginas_cuento})")

        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=filename,
            headers={
                "X-Total-Pages": str(total_paginas),
                "X-Cuento-Pages": str(paginas_cuento),
                "X-Has-Portada": str(tiene_portada),
                "X-Files-Combined": str(len(request.rutas_archivos))
            }
        )

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 🆕 ENDPOINT: COMBINAR HOJAS CUADRADAS
# ============================================================================

@app.post("/combinar-hojas-cuadradas")
async def combinar_hojas_cuadradas(data: dict):
    """
    Combina arrays separados de imágenes y textos en un PDF intercalado.

    Body JSON:
    {
        "rutas_images": ["/tmp/img1.png", "/tmp/img2.png"],
        "rutas_textos": ["/tmp/txt1.png", "/tmp/txt2.png"],
        "portada": "base64_string_opcional",
        "titulo": "Mi Cuento"
    }

    Orden final: [Portada], Imagen1, Texto1, Imagen2, Texto2, ...
    """
    logger.info(f"🔗 INICIO COMBINAR HOJAS CUADRADAS")
    logger.info(f"🚀 ENDPOINT EJECUTÁNDOSE CORRECTAMENTE")
    logger.info(f"🔍 Datos recibidos: {data}")

    # Extraer datos del dict (como el test que funciona)
    rutas_images = data.get('rutas_images', [])
    rutas_textos = data.get('rutas_textos', [])
    portada = data.get('portada', '')
    titulo = data.get('titulo', '')

    logger.info(f"📊 Imágenes: {len(rutas_images)}, Textos: {len(rutas_textos)}")
    logger.info(f"📖 Tiene portada: {bool(portada)}")
    logger.info(f"🔍 Título: '{titulo}'")

    try:
        # ============ VALIDACIÓN INICIAL ============
        logger.info("🔍 Step 1: Validación inicial...")

        if not rutas_images and not rutas_textos:
            logger.error("❌ No se proporcionaron imágenes ni textos")
            raise HTTPException(status_code=400, detail="No se proporcionaron imágenes ni textos")

        if len(rutas_images) != len(rutas_textos):
            logger.error(f"❌ Desbalance: {len(rutas_images)} imágenes vs {len(rutas_textos)} textos")
            raise HTTPException(status_code=400, detail="Debe haber el mismo número de imágenes y textos")

        num_pares = len(rutas_images)
        imagenes_combinadas = []
        logger.info(f"✅ Validación inicial OK - {num_pares} pares a procesar")

        # ============ AGREGAR PORTADA PRIMERO SI EXISTE ============
        logger.info("🔍 Step 2: Procesando portada...")
        if portada:
            try:
                logger.info("📖 Decodificando portada base64...")
                titulo_para_portada = titulo or ""
                # Decodificar imagen que YA viene lista desde crear-portada-cuadrada
                import base64
                portada_bytes = base64.b64decode(portada)
                portada_img = Image.open(io.BytesIO(portada_bytes))
                if portada_img.mode != 'RGB':
                    portada_img = portada_img.convert('RGB')

                logger.info(f"📐 DEBUG: Portada recibida (ya procesada): {portada_img.width}x{portada_img.height}")

                # Redimensionar portada a formato Amazon KDP cuadrado: 8.5" x 8.5" = 2550x2550px @ 300 DPI
                kdp_size = 2550

                # USAR EL MISMO MÉTODO QUE LAS PÁGINAS INTERNAS: RESIZE DIRECTO
                logger.info(f"📐 DEBUG: Portada original: {portada_img.width}x{portada_img.height}")

                # Redimensionar DIRECTAMENTE a formato Amazon KDP como las páginas internas
                canvas_cuadrado = portada_img.resize((kdp_size, kdp_size), Image.Resampling.LANCZOS)

                logger.info(f"📐 DEBUG: Portada redimensionada DIRECTAMENTE a {kdp_size}x{kdp_size} (igual que páginas internas)")
                logger.info(f"📐 DEBUG: Uso del espacio: 100.0% x 100.0% (llenado completo)")

                imagenes_combinadas.append(canvas_cuadrado)
                logger.info(f"✅ Portada agregada y redimensionada a {kdp_size}x{kdp_size}px (Amazon KDP)")
            except Exception as e:
                logger.error(f"❌ Error procesando portada: {e}")
                # Continuar sin portada si hay error
        else:
            logger.info("ℹ️ Sin portada - continuando...")

        # ============ PROCESAR IMÁGENES Y TEXTOS INTERCALADOS ============
        logger.info("🔍 Step 3: Procesando imágenes y textos intercalados...")
        logger.info(f"🔍 Total pares a procesar: {num_pares}")
        logger.info(f"🔍 Imágenes: {len(rutas_images)}, Textos: {len(rutas_textos)}")

        # Procesar cada par: imagen + texto
        for i in range(num_pares):
            # ============ PROCESAR IMAGEN ============
            if i < len(rutas_images):
                ruta_imagen = rutas_images[i]
                logger.info(f"🖼️ Procesando imagen {i+1}/{num_pares}: {ruta_imagen}")

                try:
                    if not os.path.exists(ruta_imagen):
                        logger.warning(f"⚠️ Imagen no encontrada: {ruta_imagen}")
                    else:
                        # Cargar imagen real y redimensionar a Amazon KDP
                        imagen_real = Image.open(ruta_imagen).convert('RGB')

                        # Redimensionar a formato Amazon KDP: 8.5" x 8.5" = 2550x2550px @ 300 DPI
                        kdp_size = 2550
                        imagen_kdp = imagen_real.resize((kdp_size, kdp_size), Image.Resampling.LANCZOS)
                        imagenes_combinadas.append(imagen_kdp)
                        logger.info(f"✅ Imagen {i+1} cargada y redimensionada a {kdp_size}x{kdp_size}px (Amazon KDP)")

                except Exception as e:
                    logger.error(f"❌ Error cargando imagen {ruta_imagen}: {e}")

            # ============ PROCESAR TEXTO ============
            if i < len(rutas_textos):
                ruta_texto = rutas_textos[i]
                logger.info(f"📄 Procesando texto {i+1}/{num_pares}: {ruta_texto}")

                try:
                    if not os.path.exists(ruta_texto):
                        logger.warning(f"⚠️ Texto no encontrado: {ruta_texto}")
                    else:
                        # Cargar página de texto y redimensionar a Amazon KDP
                        texto_real = Image.open(ruta_texto).convert('RGB')

                        # Redimensionar a formato Amazon KDP: 8.5" x 8.5" = 2550x2550px @ 300 DPI
                        kdp_size = 2550
                        texto_kdp = texto_real.resize((kdp_size, kdp_size), Image.Resampling.LANCZOS)
                        imagenes_combinadas.append(texto_kdp)
                        logger.info(f"✅ Texto {i+1} cargado y redimensionado a {kdp_size}x{kdp_size}px (Amazon KDP)")

                except Exception as e:
                    logger.error(f"❌ Error cargando texto {ruta_texto}: {e}")

            logger.info(f"📊 Total páginas hasta ahora: {len(imagenes_combinadas)}")

        logger.info(f"🔍 RESUMEN: Total páginas agregadas: {len(imagenes_combinadas)}")
        logger.info(f"🔍 Esperado: {1 if portada else 0} portada + {num_pares * 2} páginas = {(1 if portada else 0) + (num_pares * 2)}")

        # ============ VALIDACIÓN FINAL ============
        logger.info("🔍 Step 4: Validación final...")
        if not imagenes_combinadas:
            logger.error("❌ No hay imágenes válidas para combinar")
            raise HTTPException(status_code=400, detail="No hay imágenes válidas para combinar")

        logger.info(f"✅ {len(imagenes_combinadas)} imágenes listas para combinar")

        # ============ CREAR PDF COMBINADO ============
        logger.info("🔍 Step 5: Creando PDF...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        titulo_sanitizado = sanitize_filename(titulo) if titulo else "Cuento_Cuadrado"
        filename = f"{titulo_sanitizado}_{len(imagenes_combinadas)}pag_{timestamp}.pdf"
        output_path = f"/tmp/{filename}"

        logger.info(f"📁 Archivo destino: {output_path}")
        logger.info("🔄 Iniciando conversión a PDF...")

        imagenes_a_pdf(imagenes_combinadas, output_path)

        logger.info("✅ PDF creado exitosamente")

        # ============ PREPARAR RESPUESTA ============
        logger.info("🔍 Step 6: Preparando respuesta...")
        total_paginas = len(imagenes_combinadas)
        paginas_hojas = num_pares * 2
        tiene_portada = 1 if portada else 0

        logger.info(f"📊 Estadísticas finales:")
        logger.info(f"   - Total páginas: {total_paginas}")
        logger.info(f"   - Pares procesados: {num_pares}")
        logger.info(f"   - Tiene portada: {bool(tiene_portada)}")

        # Verificar que el archivo se creó correctamente
        if not os.path.exists(output_path):
            logger.error(f"❌ El archivo PDF no fue creado: {output_path}")
            raise HTTPException(status_code=500, detail="Error creando archivo PDF")

        file_size = os.path.getsize(output_path)
        logger.info(f"📄 Archivo creado: {filename} ({file_size} bytes)")

        logger.info("✅ COMBINAR HOJAS CUADRADAS COMPLETADO EXITOSAMENTE")

        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=filename,
            headers={
                "X-Total-Pages": str(total_paginas),
                "X-Hojas-Pares": str(num_pares),
                "X-Has-Portada": str(tiene_portada),
                "X-Titulo": titulo or "Sin_Titulo",
                "X-File-Size": str(file_size)
            }
        )

    except HTTPException as he:
        # Re-lanzar HTTPException sin modificar
        logger.error(f"❌ HTTPException en combinar-hojas-cuadradas: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"❌ ERROR CRÍTICO en combinar-hojas-cuadradas:")
        logger.error(f"   - Tipo: {type(e).__name__}")
        logger.error(f"   - Mensaje: {str(e)}")
        import traceback
        logger.error(f"   - Stack trace completo:")
        logger.error(traceback.format_exc())

        # Crear respuesta de error más detallada
        error_detail = f"Error interno del servidor: {type(e).__name__}: {str(e)}"
        logger.error(f"❌ Enviando error 500: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)
def crear_portada_cuadrada_con_titulo(portada_img: Image.Image, titulo: str = "Mi Cuento", tamano: int = 1200) -> Image.Image:
    """Crea una portada CUADRADA hermosa desde imagen PIL con título multilínea y auto-size."""

    logger.info(f"🔍 DEBUG en crear_portada_cuadrada_con_titulo: Título recibido: '{titulo}', Tamaño: {tamano}x{tamano}")

    # ============ USAR LA IMAGEN TAL COMO VIENE (ya procesada) ============
    # Asumir que la imagen ya viene en el tamaño correcto desde crear_portada_cuadrada_desde_base64
    # Solo redimensionar si es necesario para que llene mejor el espacio

    canvas = Image.new('RGB', (tamano, tamano), (255, 255, 255))

    # USAR MISMO MÉTODO QUE PÁGINAS INTERNAS: RESIZE DIRECTO AL TAMAÑO COMPLETO
    # Redimensionar directamente al tamaño del canvas para llenar completamente
    portada_copy = portada_img.resize((tamano, tamano), Image.Resampling.LANCZOS)

    # Pegar sin offset (llena toda la imagen)
    canvas.paste(portada_copy, (0, 0))

    logger.info(f"📐 Imagen original: {portada_img.width}x{portada_img.height}")
    logger.info(f"📐 Imagen final: {portada_copy.width}x{portada_copy.height}")
    logger.info(f"📐 Canvas: {tamano}x{tamano}, Uso: {(portada_copy.width/tamano)*100:.1f}%")

    # Canvas ya está creado arriba con la imagen centrada

    draw = ImageDraw.Draw(canvas)

    # ====================== FUENTES PROPORCIONALES ======================
    try:
        base_font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
        # Tamaño de fuente proporcional al tamaño del canvas
        font_size = max(60, tamano // 20)  # Entre 60px mínimo y proporcional
        font_default = ImageFont.truetype(base_font, font_size)
        logger.info(f"✅ Fuente cargada: tamaño {font_size}px para canvas {tamano}x{tamano}")
    except Exception as e:
        font_default = ImageFont.load_default()
        font_size = 60
        logger.warning(f"⚠️ Usando fuente por defecto: {e}")

    titulo_capitalizado = to_title_case(titulo)

    if not titulo or not titulo.strip():
        logger.info("❌ No hay título válido. Devolviendo portada sin texto.")
        return canvas.convert('RGB')

    # ============ CONFIGURAR ÁREA DE TEXTO CUADRADA ============
    margin_horizontal = tamano * 0.08  # 8% margen horizontal
    margin_vertical = tamano * 0.15    # 15% margen vertical (más espacio arriba/abajo)

    max_width = tamano - (2 * margin_horizontal)
    area_texto_y_start = margin_vertical
    area_texto_y_end = tamano - margin_vertical

    # ============ DIVIDIR TÍTULO EN LÍNEAS ============
    palabras = titulo_capitalizado.split()
    lineas_titulo = []
    linea_actual = []

    for palabra in palabras:
        test_line = ' '.join(linea_actual + [palabra])
        try:
            ancho_test = draw.textlength(test_line, font=font_default)
        except AttributeError:
            bbox = draw.textbbox((0, 0), test_line, font=font_default)
            ancho_test = bbox[2] - bbox[0]

        if ancho_test <= max_width:
            linea_actual.append(palabra)
        else:
            if linea_actual:
                lineas_titulo.append(' '.join(linea_actual))
            linea_actual = [palabra]

    if linea_actual:
        lineas_titulo.append(' '.join(linea_actual))

    if not lineas_titulo:
        logger.warning("⚠️ No se pudieron crear líneas de título")
        return canvas.convert('RGB')

    # ============ CALCULAR POSICIONAMIENTO VERTICAL ============
    try:
        altura_linea = draw.textbbox((0, 0), "Ag", font=font_default)[3] - draw.textbbox((0, 0), "Ag", font=font_default)[1]
    except:
        altura_linea = font_size

    interlineado = altura_linea * 0.3
    altura_total_texto = len(lineas_titulo) * altura_linea + (len(lineas_titulo) - 1) * interlineado
    area_disponible = area_texto_y_end - area_texto_y_start

    # Si no cabe, reducir tamaño de fuente
    while altura_total_texto > area_disponible and font_size > 30:
        font_size -= 5
        try:
            font_default = ImageFont.truetype(base_font, font_size)
        except:
            font_default = ImageFont.load_default()

        altura_linea = draw.textbbox((0, 0), "Ag", font=font_default)[3] - draw.textbbox((0, 0), "Ag", font=font_default)[1]
        interlineado = altura_linea * 0.3
        altura_total_texto = len(lineas_titulo) * altura_linea + (len(lineas_titulo) - 1) * interlineado

    # Centrar verticalmente
    y_start = area_texto_y_start + (area_disponible - altura_total_texto) // 2

    # ============ DIBUJAR TÍTULO CON EFECTOS ============
    color_dorado = "#FFD700"
    color_sombra = "#8B4513"

    y_actual = y_start
    for linea in lineas_titulo:
        try:
            ancho_linea = draw.textlength(linea, font=font_default)
        except AttributeError:
            bbox = draw.textbbox((0, 0), linea, font=font_default)
            ancho_linea = bbox[2] - bbox[0]

        x_centrado = (tamano - ancho_linea) // 2

        # Sombra (offset de 3px)
        draw.text((x_centrado + 3, y_actual + 3), linea, font=font_default, fill=color_sombra)
        # Texto principal dorado
        draw.text((x_centrado, y_actual), linea, font=font_default, fill=color_dorado)

        y_actual += altura_linea + interlineado

    logger.info(f"✅ Portada cuadrada creada: {len(lineas_titulo)} líneas, fuente {font_size}px")
    return canvas.convert('RGB')

def crear_portada_con_titulo_desde_imagen(portada_img: Image.Image, titulo: str = "Mi Cuento") -> Image.Image:
    """Crea una portada hermosa desde imagen PIL con título multilínea y auto-size."""

    logger.info(f"🔍 DEBUG en crear_portada_con_titulo_desde_imagen: Título recibido: '{titulo}'")

    # Dimensiones A4
    a4_width = 2480
    a4_height = 3508

    # ============ AJUSTAR IMAGEN A TAMAÑO A4 ============
    portada_aspect = portada_img.width / portada_img.height
    page_aspect = a4_width / a4_height

    if portada_aspect < page_aspect:
        new_width = a4_width
        new_height = int(a4_width / portada_aspect)
    else:
        new_height = a4_height
        new_width = int(a4_height * portada_aspect)

    portada_img_resized = portada_img.resize((new_width, new_height), Image.LANCZOS)

    canvas = Image.new('RGB', (a4_width, a4_height), (255, 255, 255))
    offset_x = (a4_width - new_width) // 2
    offset_y = (a4_height - new_height) // 2
    canvas.paste(portada_img_resized, (offset_x, offset_y))

    draw = ImageDraw.Draw(canvas)

    # ====================== FUENTES ======================
    try:
        base_font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
        font_default = ImageFont.truetype(base_font, 140)
    except:
        font_default = ImageFont.load_default()

    titulo_capitalizado = to_title_case(titulo)

    if not titulo or not titulo.strip():
        logger.info("❌ No hay título válido. Devolviendo portada sin texto.")
        return canvas

    # ==================== AUTO-SIZE + MULTILINE ====================
    max_width = int(a4_width * 0.92)  # Aumentado de 85% a 92% para más palabras
    max_lines = 3

    # ✔ LETRA A LA MITAD
    font_size = 110   # antes 220 → ahora la mitad

    while font_size > 30:
        try:
            font_titulo = ImageFont.truetype(base_font, font_size)
        except:
            font_titulo = font_default

        words = titulo_capitalizado.split()
        lines = []
        actual = ""

        for w in words:
            prueba = (actual + " " + w).strip()
            if draw.textlength(prueba, font=font_titulo) <= max_width:
                actual = prueba
            else:
                lines.append(actual)
                actual = w

        if actual:
            lines.append(actual)

        if len(lines) <= max_lines:
            break

        font_size -= 5

    # ==================== CALCULAR ALTURA TOTAL ====================
    line_heights = [
        draw.textbbox((0, 0), line, font=font_titulo)[3] -
        draw.textbbox((0, 0), line, font=font_titulo)[1]
        for line in lines
    ]

    total_height = sum(line_heights) + (len(lines) - 1) * 18
    titulo_y = int(a4_height - a4_height * 0.25 - total_height / 2)

    # ==================== FONDO NEGRO ====================
    fondo_x1 = int(a4_width * 0.04)  # Reducido de 5% a 4%
    fondo_x2 = int(a4_width * 0.96)  # Aumentado de 95% a 96%
    fondo_y1 = titulo_y - 30         # Reducido de 40 a 30
    fondo_y2 = titulo_y + total_height + 30  # Reducido de 40 a 30

    overlay = Image.new('RGBA', (a4_width, a4_height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([fondo_x1, fondo_y1, fondo_x2, fondo_y2], fill=(0, 0, 0, 130))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # ==================== DIBUJAR TEXTO LÍNEA POR LÍNEA ====================
    current_y = titulo_y

    # Colores dorados fuertes
    dorado_oscuro = "#8B7500"
    dorado_medio = "#DAA520"
    dorado_brillante = "#FFD700"
    dorado_luz = "#FFF4B0"

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_titulo)
        w_line = bbox[2] - bbox[0]
        h_line = bbox[3] - bbox[1]
        x_line = (a4_width - w_line) // 2

        # --- ✔ Borde oscuro para contraste ---
        for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3)]:
            draw.text((x_line + dx, current_y + dy), line, font=font_titulo, fill="#2A1E00")

        # --- ✔ Sombra suave ---
        draw.text((x_line + 2, current_y + 2), line, font=font_titulo, fill="#1A1200")

        # --- ✔ Gradiente dorado real ---
        gradiente = [dorado_oscuro, dorado_medio, dorado_brillante, dorado_luz]
        for i, color in enumerate(gradiente):
            draw.text((x_line, current_y - i), line, font=font_titulo, fill=color)

        # --- Eliminado: blanco brillante, porque hacía que el color parezca blanco ---
        # draw.text((x_line, current_y), line, font=font_titulo, fill="#FFFFFF")

        current_y += h_line + 18

    logger.info(f"✨ Portada generada con {len(lines)} líneas y tamaño de fuente {font_size}")

    return canvas.convert("RGB")



# ============================================================================
# ENDPOINT: CREAR PORTADA
# ============================================================================

@app.post("/crear-portada")
async def crear_portada(
    portada: UploadFile = File(...),  # Archivo binario de imagen (n8n Binary File)
    titulo: str = Form(...)           # Título del cuento para la portada
):
    """
    Crea una portada con título dorado desde archivo binario.
    Form data:
    - portada: archivo binario (n8n Binary File)
    - titulo: Mi Hermoso Cuento
    """
    logger.info(f"🎨 CREAR PORTADA: '{titulo[:30]}...' con archivo: {portada.filename}")
    logger.info(f"🔍 DEBUG: Título recibido en crear-portada: '{titulo}'")

    try:
        if not portada:
            raise HTTPException(status_code=400, detail="Imagen de portada requerida")

        if not titulo or not titulo.strip():
            raise HTTPException(status_code=400, detail="Título requerido")

        # Leer archivo binario y crear imagen
        portada_bytes = await portada.read()
        portada_img = Image.open(io.BytesIO(portada_bytes))

        if portada_img.mode != 'RGB':
            portada_img = portada_img.convert('RGB')

        # Crear la portada con título
        portada_final = crear_portada_con_titulo_desde_imagen(portada_img, titulo)

        # Guardar como archivo temporal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        titulo_sanitizado = sanitize_filename(titulo)
        filename = f"Portada_{titulo_sanitizado}_{timestamp}.png"
        output_path = f"/tmp/{filename}"

        # GUARDAR LA IMAGEN CORRECTA
        portada_final.save(output_path, "PNG", quality=95)

        logger.info(f"✅ Portada creada: {filename}")

        return FileResponse(
            output_path,
            media_type="image/png",
            filename=filename,
            headers={
                "X-Titulo": titulo,
                "X-Generated": str(timestamp)
            }
        )

    except Exception as e:
        logger.error(f"❌ Error creando portada: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 🆕 ENDPOINT: CREAR PORTADA CUADRADA
# ============================================================================

@app.post("/crear-portada-cuadrada")
async def crear_portada_cuadrada(
    portada: UploadFile = File(...),  # Archivo binario de imagen
    titulo: str = Form(...),          # Título del cuento
    tamano: int = Form(default=1200)  # Tamaño cuadrado (por defecto 1200x1200)
):
    """
    Crea una portada CUADRADA con título dorado para libros infantiles.

    Form data:
    - portada: archivo binario de imagen
    - titulo: Mi Hermoso Cuento
    - tamano: 1200 (opcional, para páginas cuadradas)
    """
    logger.info(f"🎨 CREAR PORTADA CUADRADA: '{titulo[:30]}...' {tamano}x{tamano}px")
    logger.info(f"🔍 DEBUG: Título recibido: '{titulo}'")

    try:
        if not portada:
            raise HTTPException(status_code=400, detail="Imagen de portada requerida")

        if not titulo or not titulo.strip():
            raise HTTPException(status_code=400, detail="Título requerido")

        # Validar tamaño
        if tamano < 400 or tamano > 3000:
            raise HTTPException(status_code=400, detail="Tamaño debe estar entre 400 y 3000 píxeles")

        # Leer archivo binario y crear imagen
        portada_bytes = await portada.read()
        portada_img = Image.open(io.BytesIO(portada_bytes))

        if portada_img.mode != 'RGB':
            portada_img = portada_img.convert('RGB')

        # Crear portada cuadrada con título
        portada_final = crear_portada_cuadrada_con_titulo(portada_img, titulo, tamano)

        # Guardar como archivo temporal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        titulo_sanitizado = sanitize_filename(titulo)
        filename = f"Portada_Cuadrada_{titulo_sanitizado}_{tamano}px_{timestamp}.png"
        output_path = f"/tmp/{filename}"

        # Guardar la imagen cuadrada
        portada_final.save(output_path, "PNG", quality=95, dpi=(300, 300))

        logger.info(f"✅ Portada cuadrada creada: {filename}")

        return FileResponse(
            output_path,
            media_type="image/png",
            filename=filename,
            headers={
                "X-Titulo": titulo,
                "X-Tamano": str(tamano),
                "X-Generated": str(timestamp),
                "X-Tipo": "portada_cuadrada",
                "ruta_portada_cuadrada": output_path
            }
        )

    except Exception as e:
        logger.error(f"❌ Error creando portada cuadrada: {str(e)}")
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
    imagen_personaje: UploadFile = File(None),  # OPCIONAL - ya no se usa
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
        # Procesar imagen fondo (YA INCLUYE TODO: fondo + personajes integrados)
        fondo_bytes = await imagen_fondo.read()
        fondo_img = Image.open(io.BytesIO(fondo_bytes))
        if fondo_img.mode != 'RGB':
            fondo_img = fondo_img.convert('RGB')

        # PERSONAJE OPCIONAL - Solo procesar si se envía (para compatibilidad)
        personaje_img = None
        if imagen_personaje is not None:
            personaje_bytes = await imagen_personaje.read()
            personaje_img = Image.open(io.BytesIO(personaje_bytes))
            personaje_img = remover_fondo_blanco(personaje_img)

        # ============ ELEGIR TIPO DE COMPOSICIÓN ============
        a4_width = 2480
        a4_height = 3508

        if tipo_composicion == "fondo_completo":
            # 🎬 ESTILO FONDO COMPLETO - Página completa épica
            logger.info(f"🎬 MODO FONDO_COMPLETO - Página {numero_pagina} (es_primera: {es_primera_pagina})")

            # Crear fondo completo SIN PERSONAJE - La imagen ya viene integrada desde n8n
            canvas = crear_fondo_solo_imagen(fondo_img, a4_width, a4_height)
            draw = ImageDraw.Draw(canvas)

            # Cargar fuentes MÁS GRANDES para mejor legibilidad infantil
            try:
                # Fuentes más grandes para niños
                font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 58)  # Tamaño más compacto
                font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 58)
                font_titulo = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 120)
            except:
                try:
                    # Intentar fuentes serif/manuscritas como fallback - MÁS GRANDES
                    font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 58)  # Serif compacto
                    font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 58)
                    font_titulo = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 120)
                except:
                    font_normal = ImageFont.load_default()
                    font_bold = ImageFont.load_default()
                    font_titulo = ImageFont.load_default()

            # NO TÍTULO EN MODO FONDO_COMPLETO - La portada ya tiene el título
            logger.info(f"🚫 Modo fondo_completo: Sin título en página {numero_pagina} (título va en portada)")

            # TEXTO manuscrito con interlineado cómodo para niños
            line_spacing = 95  # INTERLINEADO GRANDE para ocupar toda la burbuja (era 75)

            # NUEVO: Ancho de texto OPTIMIZADO para más palabras
            margin_horizontal = a4_width * 0.03  # 3% margen cada lado (reducido de 5%)
            bubble_padding_text = 25  # PADDING MÍNIMO para máximo espacio de texto
            max_width_texto = (a4_width * 0.94) - (bubble_padding_text * 2)  # 94% ancho total

            # ============ POSICIONAMIENTO OPTIMIZADO PARA 7 LÍNEAS ============
            # Texto BAJADO para dar máximo protagonismo a la imagen del cuento
            margen_inferior_fijo = 8  # MARGEN FIJO para todas las páginas
            y_start_fijo = 2700  # Y MÁS BAJA para 7 líneas (era 2200)
            y_end_fijo = 3508 - margen_inferior_fijo  # 3500px FIJO para todas

            zona_texto = {
                'x_start': a4_width * 0.05, # FIJO: 124px (menos margen)
                'x_end': a4_width * 0.95,   # FIJO: 2356px (menos margen)
                'y_start': y_start_fijo,    # FIJO: 2200px
                'y_end': y_end_fijo,        # FIJO: 3500px
                'nombre': 'inferior-optimizado'
            }

            # Configurar área de texto
            texto_x_start = zona_texto['x_start']
            texto_x_end = zona_texto['x_end']
            texto_y_start = zona_texto['y_start']
            texto_y_end = zona_texto['y_end']
            # max_width_texto ya está calculado arriba correctamente para la burbuja del 80%

            logger.info(f"📝 PÁGINA {numero_pagina}: Zona {zona_texto['nombre']}")
            logger.info(f"📐 FIJO: X({texto_x_start:.0f}-{texto_x_end:.0f}) Y({texto_y_start:.0f}-{texto_y_end:.0f}) | Margen: {margen_inferior_fijo}px")
            logger.info(f"🎯 VALORES ABSOLUTOS: y_start={y_start_fijo}, y_end={y_end_fijo}, altura_disponible={y_end_fijo-y_start_fijo}px")

            # ============ CREAR UNA BURBUJA GRANDE PARA TODO EL TEXTO ============

            # Procesar todo el texto y dividirlo en líneas
            paragrafos = texto_cuento.strip().split('\n\n')
            todas_las_lineas = []
            # LÍMITE INFORMATIVO: La IA ya dividió el texto perfectamente por páginas
            max_lines_recomendado = 7  # INFORMATIVO: 7 líneas recomendadas para mejor legibilidad

            for parrafo in paragrafos:
                palabras = parrafo.split()
                linea_actual = []

                for palabra in palabras:
                    test_line = ' '.join(linea_actual + [palabra])
                    try:
                        test_width = draw.textlength(test_line, font=font_normal)
                    except AttributeError:
                        bbox = draw.textbbox((0, 0), test_line, font=font_normal)
                        test_width = bbox[2] - bbox[0]

                    if test_width <= max_width_texto * 0.98:  # Usar 98% del ancho para menos líneas
                        linea_actual.append(palabra)
                    else:
                        if linea_actual:
                            todas_las_lineas.append(' '.join(linea_actual))
                        linea_actual = [palabra]

                # Añadir línea final del párrafo
                if linea_actual:
                    todas_las_lineas.append(' '.join(linea_actual))

            # NO RECORTAR - La IA ya dividió perfectamente el contenido
            lineas_totales = len(todas_las_lineas)

            palabras_totales = len(' '.join(todas_las_lineas).split())
            logger.info(f"📝 TEXTO IA: {palabras_totales} palabras → {lineas_totales} líneas (recomendado: {max_lines_recomendado})")

            if palabras_totales > 90:
                logger.info(f"💡 INFO: {palabras_totales} palabras (recomendado: máx 90 para 7 líneas)")

            if lineas_totales > max_lines_recomendado:
                logger.info(f"💡 INFO: {lineas_totales} líneas (recomendado: {max_lines_recomendado} líneas)")

            # ============ CREAR BURBUJA GRANDE ARMONIOSAMENTE ANCHA ============
            if todas_las_lineas:
                # Calcular dimensiones de la burbuja - ANCHO OPTIMIZADO 90% DE LA HOJA
                bubble_padding = 25  # PADDING MÍNIMO para máximo espacio de texto
                bubble_radius = 35   # Esquinas más redondeadas estilo burbuja de diálogo

                # ANCHO OPTIMIZADO: 94% del ancho de la página (6% márgenes total = 3% cada lado)
                margin_horizontal_burbuja = a4_width * 0.03  # 3% margen cada lado (reducido)
                bubble_width_total = a4_width * 0.94  # 94% de ancho de hoja

                # Altura basada en líneas de texto
                bubble_height = (len(todas_las_lineas) * line_spacing) + (bubble_padding * 2)

                # Posición de la burbuja - CENTRADA HORIZONTALMENTE CON MÁRGENES REDUCIDOS
                bubble_x = margin_horizontal_burbuja
                bubble_y = texto_y_start - bubble_padding

                # Crear burbuja semitransparente con ancho armónico
                bubble_img = Image.new('RGBA', (int(bubble_width_total), int(bubble_height)), (0, 0, 0, 0))
                bubble_draw = ImageDraw.Draw(bubble_img)

                # Fondo blanco CON MÁS OPACIDAD para lectura en fondos claros
                bubble_draw.rounded_rectangle(
                    [0, 0, bubble_width_total, bubble_height],
                    radius=bubble_radius,
                    fill=(255, 255, 255, 200),  # Blanco con MÁS OPACIDAD (78% para lectura clara)
                    outline=(100, 100, 100, 255),  # Borde gris SÓLIDO sin transparencia
                    width=3  # Borde más visible
                )

                # Pegar burbuja al canvas principal
                try:
                    canvas.paste(bubble_img, (int(bubble_x), int(bubble_y)), bubble_img)
                except:
                    # Fallback simple con más opacidad para legibilidad
                    draw.rounded_rectangle(
                        [bubble_x, bubble_y, bubble_x + bubble_width_total, bubble_y + bubble_height],
                        radius=bubble_radius,
                        fill=(255, 255, 255, 200),
                        outline=(100, 100, 100, 255),
                        width=3
                    )

                # ============ DIBUJAR TEXTO NEGRO ELEGANTE CENTRADO ============
                # Centrar texto dentro de la burbuja optimizada con padding reducido
                texto_x_centrado = margin_horizontal_burbuja + bubble_padding
                current_y = texto_y_start

                for linea in todas_las_lineas:
                    if linea.strip():
                        draw.text((texto_x_centrado, current_y), linea, font=font_normal, fill='#2c2c2c')
                        current_y += line_spacing

                logger.info(f"💬 Burbuja creada: {len(todas_las_lineas)} líneas en una sola burbuja")

            # Número de página
            if total_paginas > 1:
                page_text = f"{numero_pagina}"
                draw_texto_con_sombra_blanca(draw, a4_width - 200, a4_height - 150, page_text, font_bold, '#FFD700')

            pagina_img = canvas

        else:
            # 🎨 ESTILO HEADER+TEXTO ORIGINAL (por defecto)
            logger.info(f"🎨 MODO HEADER+TEXTO - Página {numero_pagina} (es_primera: {es_primera_pagina})")

            # Si hay personaje, combinar fondo + personaje, si no, solo fondo
            if personaje_img is not None:
                header_img = combinar_fondo_personaje(fondo_img, personaje_img, header_height, numero_pagina, total_paginas)
            else:
                # Solo fondo para el header (sin personaje)
                target_aspect = 2480 / header_height
                image_aspect = fondo_img.width / fondo_img.height

                if image_aspect < target_aspect:
                    new_width = 2480
                    new_height = int(2480 / image_aspect)
                    header_img_resized = fondo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    top_crop = max(0, (new_height - header_height) // 2)
                    header_img = header_img_resized.crop((0, top_crop, new_width, top_crop + header_height))
                else:
                    new_height = header_height
                    new_width = int(header_height * image_aspect)
                    header_img_resized = fondo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    left_crop = max(0, (new_width - 2480) // 2)
                    header_img = header_img_resized.crop((left_crop, 0, left_crop + 2480, new_height))

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
        "version": "10.2-HOJAS-CUADRADAS",
        "features": ["crear_cuento_multipagina", "crear_ficha", "combinar_documentos_con_portada", "crear_ficha_cuadrada", "combinar_hojas_cuadradas", "crear_portada_cuadrada"],
        "endpoints": {
            "POST /crear-cuento-multipagina": "Crea cuentos multipágina automático (PDF)",
            "POST /crear-ficha": "Crea ficha de 1 página (PNG)",
            "POST /crear-ficha-cuadrada": "🆕 Crea fichas dobles cuadradas imagen+texto para Amazon KDP (PNG)",
            "POST /generar-pagina-imagen": "📄 Genera una página PDF/PNG a partir de imagen binaria (mismo formato que ficha-cuadrada)",
            "POST /crear-portada": "Crea portada con título desde imagen (PNG)",
            "POST /crear-portada-cuadrada": "🆕 Crea portada CUADRADA con título para libros infantiles (PNG)",
            "POST /combinar-documentos": "Combina páginas + portada opcional en PDF",
            "POST /combinar-hojas-cuadradas": "🆕 Combina pares de hojas cuadradas en PDF"
        }
    }

# ============================================================================
# 🆕 ENDPOINT: CREAR FICHA CUADRADA
# ============================================================================

@app.post("/crear-ficha-cuadrada")
async def crear_ficha_cuadrada(
    imagen_fondo: UploadFile = File(...),
    texto: str = Form(...),
    tamano: int = Form(default=1200),
    color_fondo: str = Form(default="#FFFFFF"),
    color_texto: str = Form(default="#2c2c2c")
):
    """
    Crea DOS fichas cuadradas: una con imagen (izquierda) y otra con texto (derecha) para Amazon KDP.

    Form data:
    - imagen_fondo: archivo de imagen para página izquierda
    - texto: Era una vez una pequeña niña que vivía en el bosque...
    - tamano: 1200 (opcional)
    - color_fondo: #FFFFFF (opcional)
    - color_texto: #2c2c2c (opcional)
    """
    logger.info(f"🔲 CREAR FICHAS DOBLES: imagen + {len(texto)} caracteres, {tamano}x{tamano}px")

    try:
        if not texto or not texto.strip():
            raise HTTPException(status_code=400, detail="Texto requerido")

        if not imagen_fondo:
            raise HTTPException(status_code=400, detail="Imagen de fondo requerida")

        # Validar tamaño
        if tamano < 400 or tamano > 3000:
            raise HTTPException(status_code=400, detail="Tamaño debe estar entre 400 y 3000 píxeles")

        # ============ CREAR HOJAS SEPARADAS ============
        img_bytes = await imagen_fondo.read()
        fondo_img = Image.open(io.BytesIO(img_bytes))
        if fondo_img.mode != 'RGB':
            fondo_img = fondo_img.convert('RGB')

        # Dimensiones CUADRADAS (usar el parámetro tamano)
        logger.info(f"🔲 Usando dimensiones cuadradas: {tamano}x{tamano}px")

        # ============ HOJA 1: SOLO IMAGEN ============
        # Redimensionar imagen para llenar toda la página cuadrada
        pagina_imagen = fondo_img.resize((tamano, tamano), Image.Resampling.LANCZOS)

        # ============ HOJA 2: SOLO TEXTO ============
        # Crear página completa con solo texto
        logger.info(f"🔍 DEBUG: Creando página de texto con dimensiones {tamano}x{tamano}")
        logger.info(f"🔍 DEBUG: Texto a procesar: '{texto[:100]}...'")

        try:
            pagina_texto = crear_ficha_cuadrada_texto(
                texto=texto,
                tamano=tamano,       # Ancho cuadrado
                color_fondo=color_fondo,
                color_texto=color_texto,
                altura=tamano        # Altura cuadrada (igual al ancho)
            )
            logger.info(f"🔍 DEBUG: Página de texto creada exitosamente: {pagina_texto.size}")
        except Exception as e:
            logger.error(f"❌ ERROR creando página de texto: {e}")
            import traceback
            logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            raise

        # ============ GUARDAR AMBAS HOJAS SEPARADAS ============
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        palabras_inicio = ' '.join(texto.split()[:3])  # Primeras 3 palabras
        titulo_sanitizado = sanitize_filename(palabras_inicio)

        # Guardar hoja de imagen
        filename_imagen = f"Hoja_Imagen_{titulo_sanitizado}_{timestamp}.png"
        ruta_imagen = f"/tmp/{filename_imagen}"
        pagina_imagen.save(ruta_imagen, "PNG", quality=95, dpi=(300, 300))

        # Guardar hoja de texto
        filename_texto = f"Hoja_Texto_{titulo_sanitizado}_{timestamp}.png"
        ruta_texto = f"/tmp/{filename_texto}"
        logger.info(f"🔍 DEBUG: Intentando guardar página de texto en: {ruta_texto}")

        try:
            pagina_texto.save(ruta_texto, "PNG", quality=95, dpi=(300, 300))
            logger.info(f"✅ DEBUG: Página de texto guardada exitosamente")
        except Exception as e:
            logger.error(f"❌ ERROR guardando página de texto: {e}")
            raise

        # DEBUG: Verificar que se guardaron correctamente
        logger.info(f"🔍 DEBUG IMAGEN: Archivo guardado en: {ruta_imagen}")
        logger.info(f"🔍 DEBUG IMAGEN: Archivo existe: {os.path.exists(ruta_imagen)}")
        logger.info(f"🔍 DEBUG TEXTO: Archivo guardado en: {ruta_texto}")
        logger.info(f"🔍 DEBUG TEXTO: Archivo existe: {os.path.exists(ruta_texto)}")

        if os.path.exists(ruta_imagen):
            size = os.path.getsize(ruta_imagen)
            logger.info(f"🔍 DEBUG IMAGEN: Tamaño: {size} bytes")
        if os.path.exists(ruta_texto):
            size = os.path.getsize(ruta_texto)
            logger.info(f"🔍 DEBUG TEXTO: Tamaño: {size} bytes")

        # DEBUG: Listar archivos en /tmp para verificar
        try:
            tmp_files = os.listdir("/tmp/")
            png_files = [f for f in tmp_files if f.endswith('.png')]
            logger.info(f"🔍 DEBUG: Archivos PNG en /tmp después de guardar: {len(png_files)} archivos")
            for f in png_files[-5:]:  # Mostrar últimos 5 archivos
                logger.info(f"   📁 /tmp/{f}")
        except Exception as e:
            logger.error(f"❌ Error listando /tmp: {e}")

        palabras_totales = len(texto.split())
        logger.info(f"✅ Hojas separadas creadas: {filename_imagen} + {filename_texto} ({palabras_totales} palabras)")

        # Devolver la página de IMAGEN como principal (binario principal)
        return FileResponse(
            ruta_imagen,
            media_type="image/png",
            filename=filename_imagen,
            headers={
                "X-Tamano": str(tamano),
                "X-Palabras": str(palabras_totales),
                "X-Color-Fondo": color_fondo,
                "X-Color-Texto": color_texto,
                "ruta_ficha_cuadrada": ruta_imagen,  # Para n8n: {{ $json.headers.ruta_ficha_cuadrada }}
                "ruta_imagen": ruta_imagen,          # Ruta de la hoja de imagen
                "ruta_texto": ruta_texto,            # Ruta de la hoja de texto
                "X-Tipo": "hojas_separadas"
            }
        )

    except Exception as e:
        logger.error(f"❌ Error creando fichas dobles: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generar-pagina-imagen")
async def generar_pagina_imagen(
    imagen: UploadFile = File(...),
    tamano: int = Form(default=1200)
):
    """
    Genera una página PDF/PNG a partir de una imagen binaria.
    La imagen se expande para llenar completamente una hoja cuadrada.

    Form data:
    - imagen: archivo de imagen binaria
    - tamano: dimensión de la hoja cuadrada (default: 1200)
    """
    logger.info(f"📄 GENERAR PÁGINA DESDE IMAGEN: {tamano}x{tamano}px")

    try:
        if not imagen:
            raise HTTPException(status_code=400, detail="Imagen requerida")

        # Validar tamaño
        if tamano < 400 or tamano > 3000:
            raise HTTPException(status_code=400, detail="Tamaño debe estar entre 400 y 3000 píxeles")

        # Leer imagen binaria
        img_bytes = await imagen.read()
        source_img = Image.open(io.BytesIO(img_bytes))
        if source_img.mode != 'RGB':
            source_img = source_img.convert('RGB')

        # Redimensionar imagen para llenar toda la página cuadrada
        # (igual que en crear-ficha-cuadrada)
        pagina_imagen = source_img.resize((tamano, tamano), Image.Resampling.LANCZOS)

        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Pagina_Imagen_{timestamp}.png"
        ruta_archivo = f"/tmp/{filename}"

        # Guardar imagen
        pagina_imagen.save(ruta_archivo, "PNG", quality=95, dpi=(300, 300))

        logger.info(f"✅ Página generada: {filename} ({tamano}x{tamano}px)")

        # Devolver la página como respuesta
        return FileResponse(
            ruta_archivo,
            media_type="image/png",
            filename=filename,
            headers={
                "X-Tamano": str(tamano),
                "X-Tipo": "pagina_imagen",
                "ruta_archivo": ruta_archivo
            }
        )

    except Exception as e:
        logger.error(f"❌ Error generando página desde imagen: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "version": "10.2-HOJAS-CUADRADAS"}

@app.post("/test-combinar")
async def test_combinar(data: dict):
    """Endpoint de prueba simple para verificar conectividad."""
    logger.info(f"🧪 TEST ENDPOINT EJECUTADO")
    logger.info(f"🔍 Datos recibidos: {data}")
    return {"status": "success", "mensaje": "Endpoint funcionando", "datos_recibidos": data}
