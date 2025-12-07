from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
import io
import logging
import re
from datetime import datetime
from typing import List, Tuple
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ============================================================================
# FUNCIONES AUXILIARES (Las mismas que ya tienes)
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

# ============================================================================
# NUEVAS FUNCIONES PARA MULTIPÁGINA
# ============================================================================

def calcular_lineas_por_pagina(header_height: int, line_spacing: int = 80) -> int:
    """
    Calcula cuántas líneas de texto caben en una página.
    
    Args:
        header_height: Altura de la imagen de cabecera
        line_spacing: Espacio entre líneas
    
    Returns:
        Número aproximado de líneas que caben
    """
    a4_height = 3508
    max_height = 3380
    
    # Espacio disponible para texto
    y_text_start = header_height + 245
    espacio_disponible = max_height - y_text_start
    
    # Líneas que caben (considerando letra capital en primera página)
    lineas_aproximadas = int(espacio_disponible / line_spacing)
    
    return lineas_aproximadas

def dividir_texto_en_paginas(texto_completo: str, fonts, max_width_px: int, 
                              lineas_por_pagina: int, draw) -> List[List[Tuple[str, str]]]:
    """
    Divide el texto completo en chunks que quepan en cada página.
    
    Returns:
        Lista de páginas, donde cada página es una lista de (línea, tipo)
    """
    # Obtener todas las líneas procesadas
    todas_las_lineas = wrap_text_with_markdown(texto_completo, fonts, max_width_px, draw)
    
    paginas = []
    pagina_actual = []
    lineas_en_pagina_actual = 0
    
    # Primera página tiene menos espacio por la letra capital (3 líneas)
    lineas_primera_pagina = lineas_por_pagina - 3
    
    for linea, tipo in todas_las_lineas:
        # Determinar cuántas líneas caben en esta página
        max_lineas = lineas_primera_pagina if len(paginas) == 0 else lineas_por_pagina
        
        if tipo == 'paragraph_break':
            pagina_actual.append((linea, tipo))
            lineas_en_pagina_actual += 1  # Los párrafos cuentan como espacio
        else:
            # Si agregar esta línea excede el límite, crear nueva página
            if lineas_en_pagina_actual >= max_lineas:
                paginas.append(pagina_actual)
                pagina_actual = []
                lineas_en_pagina_actual = 0
            
            pagina_actual.append((linea, tipo))
            lineas_en_pagina_actual += 1
    
    # Agregar última página si tiene contenido
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
    """
    Crea UNA página del cuento con el diseño completo.
    
    Args:
        header_img: Imagen para la cabecera
        texto_pagina: Lista de (línea, tipo) para esta página
        titulo: Título del cuento (solo se muestra en primera página)
        es_primera_pagina: Si es la primera página (para letra capital)
        numero_pagina: Número de esta página (1-indexed)
        total_paginas: Total de páginas del cuento
        header_height: Altura de la cabecera
        estilo: Estilo visual
    
    Returns:
        Imagen PIL de la página completa
    """
    logger.info(f"📄 Creando página {numero_pagina}/{total_paginas}")
    
    a4_width = 2480
    a4_height = 3508
    
    # Inicializar canvas
    canvas = Image.new('RGBA', (a4_width, a4_height), '#FFFEF0' if estilo == "infantil" else 'white')
    
    # PROCESAMIENTO DE IMAGEN (cover centrado)
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
        
        # Capa semitransparente
        alpha_img = Image.new('RGBA', canvas.size, (255, 255, 255, 0))
        alpha_draw = ImageDraw.Draw(alpha_img)
        alpha_draw.rectangle(title_bg_rect, fill=(255, 255, 255, 180))
        canvas = Image.alpha_composite(canvas, alpha_img)
        draw = ImageDraw.Draw(canvas)
        
        # Efecto 3D en título
        title_main_color = '#E91E63'
        title_outline_color = '#8E24AA'
        outline_width = 4
        
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx * dx + dy * dy >= outline_width * outline_width:
                    draw.text((title_offset_x + dx, title_offset_y + dy), titulo_capitalizado, 
                             font=font_titulo, fill=title_outline_color)
        
        draw.text((title_offset_x, title_offset_y), titulo_capitalizado, font=font_titulo, fill=title_main_color)
    
    # Convertir a RGB
    canvas = canvas.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    
    text_color = '#2C3E50' if estilo == "infantil" else '#2c2c2c'
    
    # DIBUJAR TEXTO
    if es_primera_pagina and texto_pagina and texto_pagina[0][1] == 'text':
        # LETRA CAPITAL en primera página
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
        
        cap_y_adjustment = -15
        drop_cap_x = margin_left
        drop_cap_y_final = y_text + cap_y_adjustment
        
        cap_color = '#ef4444'
        draw.text((drop_cap_x, drop_cap_y_final), drop_cap_char, font=font_drop_cap, fill=cap_color)
        
        # Texto al lado de la letra capital
        rest_x = drop_cap_x + cap_width + 25
        rest_max_width = a4_width - rest_x - margin_right
        
        first_line_without_cap = full_first_line_content[1:].lstrip()
        
        # Re-wrap solo el primer párrafo
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        wrapped_first_para = wrap_text_with_markdown(first_line_without_cap, fonts, rest_max_width, temp_draw)
        
        y_current = y_text
        lines_beside_cap = 0
        
        # Dibujar líneas al lado de la letra capital
        for line_content, _ in wrapped_first_para:
            if lines_beside_cap < DROP_CAP_LINES:
                if line_content.strip():
                    draw_formatted_line(draw, rest_x, y_current, line_content, fonts, text_color, 
                                       max_width_px=rest_max_width)
                y_current += line_spacing
                lines_beside_cap += 1
            else:
                break
        
        # Después de la letra capital, volver al margen normal
        y_text = y_text + DROP_CAP_LINES * line_spacing + paragraph_spacing
        
        # Dibujar el resto del primer párrafo (si hay overflow)
        for i in range(lines_beside_cap, len(wrapped_first_para)):
            line_content, _ = wrapped_first_para[i]
            if y_text > max_height:
                break
            draw_formatted_line(draw, margin_left, y_text, line_content, fonts, text_color, 
                               max_width_px=max_width_px)
            y_text += line_spacing
        
        # Dibujar resto de líneas de la página (desde índice 1)
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
        # Páginas siguientes: sin letra capital
        for line, line_type in texto_pagina:
            if y_text > max_height:
                break
            
            if line_type == 'paragraph_break':
                y_text += paragraph_spacing
                continue
            
            draw_formatted_line(draw, margin_left, y_text, line, fonts, text_color, 
                               max_width_px=max_width_px)
            y_text += line_spacing
    
    # NÚMERO DE PÁGINA (abajo a la derecha)
    if total_paginas > 1:
        page_text = f"{numero_pagina}"
        bbox_page = draw.textbbox((0, 0), page_text, font=font_page_number)
        page_width = bbox_page[2] - bbox_page[0]
        page_x = a4_width - margin_right - page_width
        page_y = a4_height - 150
        
        draw.text((page_x, page_y), page_text, font=font_page_number, fill='#999999')
    
    # Borde decorativo
    if estilo == "infantil":
        draw_wavy_border(draw, a4_width, a4_height)
    
    return canvas

def imagenes_a_pdf(imagenes: List[Image.Image], output_path: str):
    """
    Convierte una lista de imágenes PIL a un PDF multipágina.
    
    Args:
        imagenes: Lista de imágenes PIL (RGB)
        output_path: Ruta donde guardar el PDF
    """
    if not imagenes:
        raise ValueError("No hay imágenes para convertir a PDF")
    
    # Convertir todas a RGB si no lo están
    imagenes_rgb = [img.convert('RGB') if img.mode != 'RGB' else img for img in imagenes]
    
    # Guardar como PDF multipágina
    imagenes_rgb[0].save(
        output_path,
        save_all=True,
        append_images=imagenes_rgb[1:],
        resolution=300.0,
        quality=95
    )
    
    logger.info(f"✅ PDF creado con {len(imagenes)} páginas: {output_path}")



# ============================================================================
# 🆕 ENDPOINT NUEVO: COMBINAR DOCUMENTOS
# ============================================================================

@app.post("/combinar-documentos")
async def combinar_documentos(rutas_archivos: List[str] = Form(...)):
    """
    Combina múltiples imágenes PNG o PDFs en un solo PDF multipágina.
    
    Args:
        rutas_archivos: Lista de rutas de archivos a combinar (PNG o PDF)
    
    Returns:
        PDF con todas las páginas combinadas
    """
    logger.info(f"🔗 COMBINAR DOCUMENTOS: {len(rutas_archivos)} archivos")
    
    try:
        if not rutas_archivos:
            raise HTTPException(status_code=400, detail="No se proporcionaron archivos para combinar")
        
        imagenes_combinadas = []
        
        for i, ruta in enumerate(rutas_archivos):
            logger.info(f"📄 Procesando archivo {i+1}/{len(rutas_archivos)}: {ruta}")
            
            # Verificar que el archivo existe
            if not os.path.exists(ruta):
                logger.warning(f"⚠️ Archivo no encontrado: {ruta}")
                continue
            
            # Determinar tipo de archivo
            extension = os.path.splitext(ruta)[1].lower()
            
            if extension in ['.png', '.jpg', '.jpeg']:
                # Cargar imagen directamente
                img = Image.open(ruta)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                imagenes_combinadas.append(img)
                
            elif extension == '.pdf':
                # Si es PDF, extraer páginas (requiere pdf2image)
                try:
                    from pdf2image import convert_from_path
                    paginas_pdf = convert_from_path(ruta, dpi=300)
                    imagenes_combinadas.extend(paginas_pdf)
                except ImportError:
                    logger.error("❌ pdf2image no está instalado. Instala con: pip install pdf2image")
                    raise HTTPException(status_code=500, detail="pdf2image no disponible")
            else:
                logger.warning(f"⚠️ Formato no soportado: {extension}")
        
        if not imagenes_combinadas:
            raise HTTPException(status_code=400, detail="No se pudieron cargar imágenes válidas")
        
        # Crear PDF combinado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Cuento_Completo_{len(imagenes_combinadas)}pag_{timestamp}.pdf"
        output_path = f"/tmp/{filename}"
        
        imagenes_a_pdf(imagenes_combinadas, output_path)
        
        logger.info(f"✅ Documentos combinados: {len(imagenes_combinadas)} páginas")
        
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=filename,
            headers={
                "X-Total-Pages": str(len(imagenes_combinadas)),
                "X-Files-Combined": str(len(rutas_archivos))
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error combinando documentos: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# ENDPOINT PRINCIPAL MULTIPÁGINA
# ============================================================================

@app.post("/crear-cuento-multipagina")
async def crear_cuento_multipagina(
    imagen: UploadFile = File(...),
    texto_cuento: str = Form(...),
    titulo: str = Form(default=""),
    header_height: int = Form(default=1150),
    estilo: str = Form(default="infantil"),
):
    """
    Crea un cuento de múltiples páginas automáticamente según la longitud del texto.
    Genera un PDF con todas las páginas.
    """
    logger.info(f"📚 CUENTO MULTIPÁGINA: {len(texto_cuento)} caracteres")
    
    try:
        # Leer imagen
        img_bytes = await imagen.read()
        header_img = Image.open(io.BytesIO(img_bytes))
        
        if header_img.mode != 'RGB':
            header_img = header_img.convert('RGB')
        
        # Configuración
        margin_left = 160
        margin_right = 160
        max_width_px = 2480 - margin_left - margin_right
        line_spacing = 80
        
        # Calcular cuántas líneas caben por página
        lineas_por_pagina = calcular_lineas_por_pagina(header_height, line_spacing)
        logger.info(f"📐 Calculadas ~{lineas_por_pagina} líneas por página")
        
        # Preparar fuentes para cálculo de wrapping
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
        
        # Dividir texto en páginas
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        paginas_texto = dividir_texto_en_paginas(
            texto_cuento, 
            fonts, 
            max_width_px, 
            lineas_por_pagina, 
            temp_draw
        )
        
        total_paginas = len(paginas_texto)
        logger.info(f"📄 Se crearán {total_paginas} páginas")
        
        # Generar cada página
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
        
        # Crear PDF con todas las páginas
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        titulo_sanitizado = sanitize_filename(titulo) if titulo else "Sin_Titulo"
        filename = f"Cuento_{titulo_sanitizado}_{total_paginas}pag_{timestamp}.pdf"
        
        output_path = f"/tmp/{filename}"
        imagenes_a_pdf(imagenes_paginas, output_path)
        
        # Calcular palabras aproximadas
        palabras_aprox = len(texto_cuento.split())
        logger.info(f"✅ Cuento creado: {total_paginas} páginas, ~{palabras_aprox} palabras")
        
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
# ENDPOINT ORIGINAL (1 PÁGINA) - MANTENIDO PARA COMPATIBILIDAD
# ============================================================================

@app.post("/crear-ficha")
async def crear_ficha(
    imagen: UploadFile = File(...),
    texto_cuento: str = Form(...),
    titulo: str = Form(default=""),
    header_height: int = Form(default=1150),
    estilo: str = Form(default="infantil"),
):
    """Endpoint original - crea UNA sola página (compatibilidad hacia atrás)"""
    logger.info(f"📥 FICHA SIMPLE (1 página): {len(texto_cuento)} chars")
    
    # Truncar texto si es muy largo (advertencia)
    palabras = len(texto_cuento.split())
    if palabras > 270:
        logger.warning(f"⚠️ Texto largo ({palabras} palabras). Considera usar /crear-cuento-multipagina")
    
    # Usar la función de crear página única
    try:
        img_bytes = await imagen.read()
        header_img = Image.open(io.BytesIO(img_bytes))
        
        if header_img.mode != 'RGB':
            header_img = header_img.convert('RGB')
        
        # Preparar texto como si fuera una sola página
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
        
        # Crear una sola página
        pagina_img = crear_pagina_cuento(
            header_img=header_img,
            texto_pagina=texto_lines,
            titulo=titulo,
            es_primera_pagina=True,
            numero_pagina=1,
            total_paginas=1,
            header_height=header_height,
            estilo=estilo
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        titulo_sanitizado = sanitize_filename(titulo) if titulo else "Sin_Titulo"
        filename = f"Cuento_{titulo_sanitizado}_ficha_lectura_{timestamp}.png"
        
        output_path = f"/tmp/{filename}"
        pagina_img.save(output_path, quality=95, dpi=(300, 300))
        
        logger.info(f"✅ Ficha simple creada: {filename}")
        
        return FileResponse(output_path, media_type="image/png", filename=filename)
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Tu endpoint de preguntas se mantiene igual...
# (Copiar tu función crear_hoja_preguntas aquí si la necesitas)

@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "8.0-MULTIPAGINA",
        "features": ["crear_cuento_multipagina", "crear_ficha", "crear_hoja_preguntas"],
        "endpoints": {
            "POST /crear-cuento-multipagina": "✨ NUEVO: Crea cuentos de 1-10+ páginas automáticamente (PDF)",
            "POST /crear-ficha": "Crea ficha de 1 página (PNG) - original",
            "POST /crear-hoja-preguntas": "Crea hoja de preguntas"
        },
        "message": "Cuento multipágina automático basado en longitud del texto"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "version": "8.0-MULTIPAGINA"}
