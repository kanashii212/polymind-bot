def generate_image(prompt: str) -> str:
    """Генерирует изображение через OpenRouter"""
    if not OPENROUTER_API_KEY:
        return "❌ OPENROUTER_API_KEY не настроен. Добавьте его в переменные окружения Render."
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://t.me/MAB_GatewayBot",
                "X-Title": "MAB Gateway Bot"
            },
            json={
                "model": "black-forest-labs/flux-1.1-pro",
                "messages": [
                    {"role": "user", "content": f"Generate an image: {prompt}"}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                return f"🖼️ Изображение по запросу: '{prompt}'\n\n{content}"
            else:
                return f"❌ Неожиданный ответ от API: {data}"
        else:
            return f"❌ Ошибка API: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "❌ Превышено время ожидания API. Попробуйте позже."
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"
