# parsing data

import pandas as pd
import stanza
stanza.download('es')

nlp = stanza.Pipeline("es", processors="tokenize,pos,ner,lemma,depparse", tokenize_pretokenized=False)

# Función para procesar cada fila de la columna 'CM' en el DataFrame
def process_text_with_stanza(dataframe, text_column):
    """Process each row at the 'CM' column in the DF with stanza in spanish
    """
    # Crear listas para almacenar los resultados
    pos_tags = []
    ner_tags = []
    lemmas = []
    dependencies = []

    # Process each texto in text_column specified
    for text in dataframe[text_column]:
        doc = nlp(text)

        # Get pos, ner, lemmas and dependencies
        pos = [(word.text, word.upos) for sentence in doc.sentences for word in sentence.words]
        ner = [(ent.text, ent.type) for ent in doc.ents]
        lemma = [(word.text, word.lemma) for sentence in doc.sentences for word in sentence.words]
        dep = [(word.text, word.head, word.deprel) for sentence in doc.sentences for word in sentence.words]

        # Save data
        pos_tags.append(pos)
        ner_tags.append(ner)
        lemmas.append(lemma)
        dependencies.append(dep)

    # Add parsing results in the df
    dataframe['POS_Tags'] = pos_tags
    dataframe['NER_Tags'] = ner_tags
    dataframe['Lemmas'] = lemmas
    dataframe['Dependencies'] = dependencies

    return dataframe


if __name__ == "__main__":
    # args parser config
    xparser = argparse.ArgumentParser(description="Parsing data from a csv file.")
    xparser.add_argument("data", help="Path to the file to preprocess")
    xparser.add_argument("text_column", help="Column name to the text column affected by the parsing")
    args = xparser.parse_args()
    # function use
    df = pd.read_csv(args.data)
    parsed_data = process_text_with_stanza(df, args.text_column)
    parsed_data.to_csv("./data/parsed_data.csv", index=False)
    print("data savec as parsed_data")
