import json
from pathlib import Path
from langchain_core.documents import Document

BASE = Path(__file__).resolve().parent
RESULTS = BASE.parent / "aula_04" / "results"

SCHEMA = {
    "fonte": ("string", "Nome do arquivo .md de origem"),
    "documento_id": ("string", "Identificador do documento"),
    "chunk_index": ("int", "Posição do chunk dentro do documento"),
    "estrategia": ("string", "Estratégia de chunking utilizada"),
    "chunk_size": ("int/null", "Tamanho configurado para o chunk"),
    "chunk_overlap": ("int", "Overlap configurado"),
    "n_caracteres": ("int", "Quantidade real de caracteres"),
    "chunk_id": ("string", "Identificador único do chunk"),
    "pagina": ("int/null", "Página original, quando disponível"),
    "secao": ("string/null", "Seção ou heading associado"),
    "n_palavras": ("int", "Quantidade aproximada de palavras"),
}

CAMPOS_PROPRIOS = {
    "chunk_id": "Permite rastrear exatamente qual chunk foi recuperado.",
    "pagina": "Permite informar a página de origem da informação.",
    "secao": "Permite identificar a seção associada ao trecho.",
    "n_palavras": "Permite analisar a granularidade textual do chunk.",
}


def criar_documentos():
    base = [
        (
            "Embeddings representam textos como vetores numéricos.",
            "conceitos_ia.md", 1, "teoria", "embeddings"
        ),
        (
            "Chunking divide documentos em partes menores para processamento.",
            "chunking.md", 2, "teoria", "chunking"
        ),
        (
            "O Recursive Splitter tenta preservar unidades naturais do texto.",
            "chunking.md", 3, "pratica", "chunking"
        ),
        (
            "RAG combina recuperação de informação com modelos generativos.",
            "rag.md", 4, "teoria", "RAG"
        ),
        (
            "Tokenização transforma texto em unidades processáveis por modelos.",
            "tokenizacao.md", 5, "teoria", "tokenizacao"
        ),
    ]

    documentos = [
        Document(
            page_content=texto,
            metadata={
                "fonte": fonte,
                "pagina": pagina,
                "tipo": tipo,
                "tema": tema,
                "autor": "Aluno",
            },
        )
        for texto, fonte, pagina, tipo, tema in base
    ]

    documentos.append(
        Document(
            page_content=(
                "Metadados ajudam a filtrar e rastrear documentos durante uma busca."
            ),
            metadata={
                "fonte": "metadados.md",
                "pagina": 6,
                "tipo": "exemplo",
                "tema": "metadados",
                "tags": ["RAG", "busca", "filtros"],
                "detalhes": {
                    "nivel": "introdutorio",
                    "revisado": True
                },
            },
        )
    )

    return documentos


def carregar_chunk_real():
    caminho = (
        RESULTS
        / "bioetica_e_ia"
        / "test_09"
        / "chunks_embeddings.json"
    )

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    chunks = json.loads(
        caminho.read_text(encoding="utf-8")
    )

    indice = min(10, len(chunks) - 1)
    chunk = chunks[indice]
    texto = chunk["text"]
    original = chunk.get("metadata", {})

    metadata = {
        "fonte": "bioetica_e_ia.md",
        "documento_id": chunk.get(
            "document_id",
            "bioetica_e_ia"
        ),
        "chunk_index": indice,
        "estrategia": chunk.get(
            "strategy",
            "recursive"
        ),
        "chunk_size": chunk.get(
            "chunk_size",
            1000
        ),
        "chunk_overlap": chunk.get(
            "chunk_overlap",
            100
        ),
        "n_caracteres": len(texto),
        "chunk_id": chunk.get(
            "chunk_id",
            f"bioetica_e_ia_test09_chunk{indice + 1:04d}"
        ),
        "pagina": original.get("page"),
        "secao": (
            original.get("section")
            or original.get("heading")
        ),
        "n_palavras": len(texto.split()),
    }

    documento = Document(
        page_content=texto,
        metadata=metadata
    )

    exemplo = {
        "page_content": documento.page_content,
        "metadata": documento.metadata
    }

    saida = (
        BASE
        / "exercicio_02_exemplo_chunk.json"
    )

    saida.write_text(
        json.dumps(
            exemplo,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return exemplo, saida


def gerar_relatorio(documentos, exemplo):
    linhas = [
        "# Relatório - Aula 05",
        "",
        "## Exercício 1 - Criando Documents",
        "",
        (
            f"Foram criados **{len(documentos)} objetos `Document`** "
            "sobre embeddings, chunking, RAG, tokenização e metadados."
        ),
        "",
        "O resultado de `len(documentos)` foi:",
        "",
        f"```text\n{len(documentos)}\n```",
        "",
        (
            "Cada `Document` possui `page_content`, responsável pelo texto, "
            "e `metadata`, responsável pelas informações associadas."
        ),
        "",
        "### Que tipos de dados são aceitos em `metadata`?",
        "",
        (
            "Nos testes realizados, `metadata` aceitou strings, números, "
            "valores booleanos, listas e dicionários aninhados."
        ),
        (
            "A lista utilizada em `tags` e o dicionário utilizado em "
            "`detalhes` foram armazenados normalmente."
        ),
        (
            "É importante observar que uma vector store pode possuir "
            "restrições próprias para os tipos de metadados utilizados "
            "em operações de filtragem."
        ),
        "",
        "### O que acontece se um `Document` for criado sem `metadata`?",
        "",
        (
            "Não ocorre erro. Quando `metadata` não é informado, "
            "o LangChain utiliza automaticamente um dicionário vazio (`{}`)."
        ),
        "",
        "## Exercício 2 - Schema de metadados",
        "",
        "| Campo | Tipo | Descrição |",
        "|---|---|---|",
    ]

    for campo, (tipo, descricao) in SCHEMA.items():
        linhas.append(
            f"| `{campo}` | {tipo} | {descricao} |"
        )

    linhas += [
        "",
        "### Campos próprios adicionados",
        "",
        (
            "- **`chunk_id`**: permite identificar unicamente o trecho recuperado.  "
        ),
        (
            "  **Pergunta que permite responder:** "
            "\"Qual chunk específico foi utilizado para recuperar esta informação?\""
        ),
        "",
        (
            "- **`pagina`**: permite localizar a informação no documento original.  "
        ),
        (
            "  **Pergunta que permite responder:** "
            "\"Em qual página do documento está essa informação?\""
        ),
        "",
        (
            "- **`secao`**: permite identificar a parte estrutural do documento "
            "à qual o trecho pertence.  "
        ),
        (
            "  **Pergunta que permite responder:** "
            "\"Em qual seção do documento esse conteúdo aparece?\""
        ),
        "",
        (
            "- **`n_palavras`**: permite analisar o tamanho textual dos chunks.  "
        ),
        (
            "  **Pergunta que permite responder:** "
            "\"Quantas palavras possui esse chunk e qual é sua granularidade "
            "em relação aos demais?\""
        ),
        "",
        "### Exemplo preenchido com um chunk real",
        "",
        (
            "O exemplo abaixo foi obtido do documento `bioetica_e_ia`, "
            "utilizando a estratégia Recursive da Aula 04."
        ),
        "",
        "```json",
        json.dumps(
            exemplo,
            ensure_ascii=False,
            indent=2
        ),
        "```",
        "",
        (
            "Os campos `pagina` e `secao` aparecem como `null` neste exemplo "
            "porque essas informações não estavam disponíveis nos metadados "
            "gerados na etapa anterior."
        ),
        "",
        "### Qual campo incluir para citar a fonte na resposta final do RAG?",
        "",
        (
            "O principal campo adicional seria `pagina`, utilizado em conjunto "
            "com `fonte`. O campo `secao` também pode complementar a referência "
            "quando estiver disponível."
        ),
        "",
        (
            "Assim, uma resposta poderia apresentar uma referência como: "
            "**Fonte: bioetica_e_ia.md, página 10, seção Discussão.**"
        ),
        "",
        (
            "No exemplo atual, `pagina` e `secao` estão como `null`, pois esses "
            "dados não foram preservados nos metadados da etapa anterior. "
            "Para uma citação exata por página em um sistema RAG futuro, essas "
            "informações precisariam ser mantidas durante a extração e o "
            "processamento do documento."
        ),
        "",
        "### Por que `chunk_index` é útil?",
        "",
        (
            "O `chunk_index` registra a posição do chunk dentro do documento."
        ),
        (
            "Se um trecho recuperado estiver cortado no meio de uma explicação, "
            "é possível buscar o chunk anterior (`chunk_index - 1`) ou o "
            "posterior (`chunk_index + 1`) para recuperar mais contexto."
        ),
        (
            "Além disso, o campo ajuda a reconstruir a ordem original dos chunks."
        ),
        "",
        "## Conclusão",
        "",
        (
            "A classe `Document` padroniza a representação dos conteúdos "
            "utilizados pelo LangChain. O texto fica em `page_content`, enquanto "
            "as informações necessárias para rastreamento, filtragem e citação "
            "ficam em `metadata`."
        ),
        (
            "Os embeddings não são armazenados dentro do `Document`, "
            "pois são responsabilidade da vector store."
        ),
    ]

    caminho = BASE / "RELATORIO.md"

    caminho.write_text(
        "\n".join(linhas),
        encoding="utf-8"
    )

    return caminho

def main():
    documentos = criar_documentos()

    print("=" * 60)
    print("EXERCÍCIO 1 - DOCUMENTS")
    print("=" * 60)

    for i, doc in enumerate(documentos, 1):
        print(f"\nDocumento {i}")
        print("page_content:", doc.page_content)
        print("metadata:", doc.metadata)

    print(
        f"\nQuantidade de documentos: "
        f"{len(documentos)}"
    )

    sem_metadata = Document(
        page_content="Documento criado sem metadados."
    )

    print(
        "\nDocumento sem metadata:",
        sem_metadata.metadata
    )

    print("\n" + "=" * 60)
    print("EXERCÍCIO 2 - SCHEMA DE METADADOS")
    print("=" * 60)

    for campo, (_, descricao) in SCHEMA.items():
        print(
            f"{campo}: {descricao}"
        )

    exemplo, json_saida = carregar_chunk_real()

    relatorio_saida = gerar_relatorio(
        documentos,
        exemplo
    )

    print(
        "\nExemplo com chunk real da Aula 04:"
    )

    print(
        json.dumps(
            exemplo,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        f"\nJSON salvo em: "
        f"{json_saida}"
    )

    print(
        f"Relatório salvo em: "
        f"{relatorio_saida}"
    )

    print(
        "\nATIVIDADE FINALIZADA COM SUCESSO"
    )


if __name__ == "__main__":
    main()