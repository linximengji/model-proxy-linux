"""Local OCR via RapidOCR (ONNX Runtime) — extract text from image blocks before routing."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

_engine = None


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    try:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    except ImportError:
        return None
    except Exception as e:
        print(f"[ocr] RapidOCR init failed: {e}", file=sys.stderr)
        return None
    return _engine


def extract_text_from_image(image_path: str) -> str:
    if not os.path.isfile(image_path):
        return ""
    engine = _get_engine()
    if engine is None:
        return ""
    try:
        result, elapse = engine(image_path)
    except Exception as e:
        print(f"[ocr] OCR failed on {image_path}: {e}", file=sys.stderr)
        return ""
    if not result:
        return ""
    lines = [text.strip() for _, text, _ in result if text.strip()]
    return "\n".join(lines)


def try_ocr_messages(messages) -> str:
    """Scan user messages for base64 image blocks, run OCR on each.

    Appends OCR text as a new text block alongside each image (does NOT remove image).
    Returns the concatenated OCR text (empty string if no images or OCR failed).
    """
    import base64
    import tempfile
    all_text = ""

    for msg in messages:
        if msg.get("role") not in ("user",):
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue

        new_content = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            if block.get("type") not in ("image", "image_url"):
                new_content.append(block)
                continue

            img_data = None
            source = block.get("source", block.get("image_url", {}))
            if isinstance(source, dict) and source.get("type") == "base64":
                img_data = base64.b64decode(source.get("data", ""))
            if not img_data:
                new_content.append(block)
                continue

            mt = source.get("media_type", "image/png")
            ext = ".webp" if mt == "image/webp" else ".jpg" if mt == "image/jpeg" else ".png"

            text = ""
            tmp = None
            try:
                fd, tmp = tempfile.mkstemp(suffix=ext)
                os.close(fd)
                with open(tmp, "wb") as f:
                    f.write(img_data)
                text = extract_text_from_image(tmp)
            finally:
                if tmp and os.path.isfile(tmp):
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass

            # Keep image, append OCR text alongside
            new_content.append(block)
            if text.strip():
                new_content.append({"type": "text", "text": f"[OCR extracted]\n{text}"})
                all_text += "\n" + text

        msg["content"] = new_content

    return all_text.strip()
