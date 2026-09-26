import ollama

# 1. Tool
def ottieni_meteo(citta: str) -> str:
    """Restituisce le condizioni meteo attuali per una data città in Italia."""
    print(f"\n[SISTEMA] Esecuzione strumento in corso: cerco il meteo per {citta}...")
    dati_meteo = {"Roma": "Soleggiato, 25°C", "Milano": "Pioggia, 15°C", "Firenze": "Nuvoloso, 20°C"}
    return dati_meteo.get(citta, "Dati meteo non disponibili per questa città.")

# 2. Domanda ambigua per il test di routing
domanda = "Ciao, come stai?"
print(f"Utente: {domanda}")
print("Elaborazione in corso...")

# 3. Prima chiamata
messages = [{'role': 'user', 'content': domanda}]
response = ollama.chat(
    model='qwen2.5:14b',
    messages=messages,
    tools=[ottieni_meteo]
)

# 4. Gestione Tool Calling
if response.get('message', {}).get('tool_calls'):
    print("\n[INFO] L'agente ha deciso che ha bisogno di uno strumento per rispondere!")
    
    # FIX CODE REVIEW: Salviamo la richiesta del tool nella cronologia
    messages.append(response['message'])
    
    for tool in response['message']['tool_calls']:
        nome_funzione = tool['function']['name']
        argomenti = tool['function']['arguments']
        
        print(f"[INFO] L'agente chiede di eseguire: {nome_funzione}({argomenti})")
        
        if nome_funzione == 'ottieni_meteo':
            citta_scelta = argomenti.get('citta', '')
            risultato_tool = ottieni_meteo(citta=citta_scelta)
            
            print(f"[SISTEMA] Risultato ottenuto dal tool: {risultato_tool}")
            
            # Aggiungiamo la risposta del tool alla cronologia
            messages.append({'role': 'tool', 'content': risultato_tool, 'name': nome_funzione})
            
            # Sintesi finale (assicurati che sia 14b)
            risposta_finale = ollama.chat(
                model='qwen2.5:14b',
                messages=messages
            )
            
            print("\n--- RISPOSTA FINALE DELL'AGENTE ---")
            print(risposta_finale['message']['content'])

else:
    print("\nL'agente ha risposto direttamente senza usare strumenti:")
    print(response['message']['content'])