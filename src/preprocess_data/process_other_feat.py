try:
    import nltk

    nltk.download("punkt")
    from nltk.tokenize import sent_tokenize
except:
    print("Try installing first with", "!python3 -m pip install", sep="\n\n")


def process_phrases(contexts, cms):
    all_phrases = []
    noise_detected = []

    for i in range(len(cms)):
        sent_tokenized = sent_tokenize(contexts[i])
        saved = False  # Inicializar la variable 'saved' para cada contexto

        if len(sent_tokenized) == 1:
            all_phrases.append(sent_tokenized[0])
            noise_detected.append(False)
            saved = True
        else:
            for j, instr in enumerate(sent_tokenized):
                if cms[i] in instr:
                    all_phrases.append(instr)
                    noise_detected.append(False)
                    saved = True
                    break  # Sale del bucle tan pronto encuentra una coincidencia

            # Si no se encontró ninguna coincidencia
            if not saved:
                # Imprimir la identificación errónea
                # print(f"{cms[i]} : {sent_tokenized}")

                # Seleccionar la frase más parecida
                best_match = ""
                for phrase in sent_tokenized:
                    if cms[i] in phrase:
                        best_match = phrase
                        break
                    for punct in [".", "!", "?"]:
                        if punct in phrase:
                            part_before_punct = phrase.split(punct)[0]
                            if cms[i] in part_before_punct:
                                best_match = part_before_punct
                                break
                    if best_match:
                        break
                if not best_match:
                    best_match = max(sent_tokenized, key=len)
                all_phrases.append(best_match)
                noise_detected.append(True)

    return all_phrases, noise_detected
