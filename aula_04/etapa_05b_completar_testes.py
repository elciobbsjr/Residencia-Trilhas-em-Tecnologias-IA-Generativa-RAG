import json
from pathlib import Path

from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from etapa_05_finalistas import (
    DOCUMENTOS,
    PASTA_RESULTS,
    EMBEDDING_MODEL,
    gerar_embeddings,
    normalizar_chunks_limite_tokens,
)


# ============================================================
# CONFIGURAÇÕES DOS 10 TESTES
# ============================================================

CONFIGURACOES = {
    1: {
        "strategy": "fixed_200",
        "chunk_size": 200,
        "chunk_overlap": 0,
    },
    2: {
        "strategy": "fixed_500",
        "chunk_size": 500,
        "chunk_overlap": 0,
    },
    3: {
        "strategy": "fixed_1000",
        "chunk_size": 1000,
        "chunk_overlap": 0,
    },
    4: {
        "strategy": "fixed_2000",
        "chunk_size": 2000,
        "chunk_overlap": 0,
    },
    5: {
        "strategy": "fixed_overlap_50",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    6: {
        "strategy": "fixed_overlap_200",
        "chunk_size": 500,
        "chunk_overlap": 200,
    },
    7: {
        "strategy": "paragraph",
        "chunk_size": None,
        "chunk_overlap": 0,
    },
    8: {
        "strategy": "three_sentences",
        "chunk_size": 3,
        "chunk_overlap": 0,
    },
    9: {
        "strategy": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 100,
    },
    10: {
        "strategy": "markdown_headers",
        "chunk_size": None,
        "chunk_overlap": 0,
    },
}


# Testes que ainda precisam estar disponíveis
# nos 12 documentos.
TESTES_COMPLEMENTARES = [
    1,
    2,
    10,
]


# ============================================================
# CARREGAR MARKDOWN
# ============================================================

def carregar_markdown(document_id):

    caminho = (
        PASTA_RESULTS
        / document_id
        / "markdown"
        / f"{document_id}.md"
    )

    if not caminho.exists():

        raise FileNotFoundError(
            f"Markdown não encontrado:\n{caminho}"
        )

    return caminho.read_text(
        encoding="utf-8"
    )


# ============================================================
# TESTE 1 E 2
# ============================================================

def chunking_fixo(
    texto,
    chunk_size
):

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    textos = splitter.split_text(
        texto
    )

    return [
        {
            "text": trecho,
            "metadata": {}
        }
        for trecho in textos
        if trecho.strip()
    ]


# ============================================================
# TESTE 10 - MARKDOWN
# ============================================================

def chunking_markdown(
    texto
):

    headers_to_split_on = [
        ("#", "heading_1"),
        ("##", "heading_2"),
        ("###", "heading_3"),
        ("####", "heading_4"),
        ("#####", "heading_5"),
        ("######", "heading_6"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    documentos = splitter.split_text(
        texto
    )

    chunks = []

    for documento in documentos:

        metadata = dict(
            documento.metadata
        )

        heading = None
        heading_level = None

        # Pega o heading mais específico.
        for nivel in range(
            6,
            0,
            -1
        ):

            chave = (
                f"heading_{nivel}"
            )

            if chave in metadata:

                heading = metadata[
                    chave
                ]

                heading_level = nivel

                break

        metadata[
            "heading"
        ] = heading

        metadata[
            "heading_level"
        ] = heading_level

        metadata[
            "section"
        ] = metadata.get(
            "heading_1"
        )

        metadata[
            "subsection"
        ] = (
            metadata.get(
                "heading_2"
            )
            or metadata.get(
                "heading_3"
            )
        )

        chunks.append({
            "text":
                documento.page_content,

            "metadata":
                metadata
        })

    return chunks


# ============================================================
# EXECUTAR CHUNKING
# ============================================================

def executar_chunking(
    texto,
    test_id
):

    if test_id == 1:

        return chunking_fixo(
            texto,
            200
        )

    if test_id == 2:

        return chunking_fixo(
            texto,
            500
        )

    if test_id == 10:

        return chunking_markdown(
            texto
        )

    raise ValueError(
        f"Teste inválido: {test_id}"
    )


# ============================================================
# NORMALIZAR SCHEMA DE UM ARQUIVO EXISTENTE
# ============================================================

def normalizar_schema_arquivo(
    caminho,
    test_id
):
    """
    Garante que os JSONs existentes também possuam
    chunk_size e chunk_overlap.
    """

    if not caminho.exists():

        return

    with caminho.open(
        "r",
        encoding="utf-8"
    ) as arquivo:

        dados = json.load(
            arquivo
        )

    configuracao = (
        CONFIGURACOES[
            test_id
        ]
    )

    alterado = False

    for chunk in dados:

        if (
            "chunk_size"
            not in chunk
        ):

            chunk[
                "chunk_size"
            ] = configuracao[
                "chunk_size"
            ]

            alterado = True

        if (
            "chunk_overlap"
            not in chunk
        ):

            chunk[
                "chunk_overlap"
            ] = configuracao[
                "chunk_overlap"
            ]

            alterado = True

    if alterado:

        with caminho.open(
            "w",
            encoding="utf-8"
        ) as arquivo:

            json.dump(
                dados,
                arquivo,
                ensure_ascii=False,
                indent=2
            )


# ============================================================
# PROCESSAR UM TESTE
# ============================================================

def processar(
    document_id,
    test_id
):

    config = CONFIGURACOES[
        test_id
    ]

    pasta = (
        PASTA_RESULTS
        / document_id
        / f"test_{test_id:02d}"
    )

    pasta.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho = (
        pasta
        / "chunks_embeddings.json"
    )

    # --------------------------------------------------------
    # JÁ EXISTE
    # --------------------------------------------------------

    if (
        caminho.exists()
        and caminho.stat().st_size > 0
    ):

        print(
            "   Resultado já existe."
        )

        normalizar_schema_arquivo(
            caminho,
            test_id
        )

        return "existing"

    # --------------------------------------------------------
    # MARKDOWN
    # --------------------------------------------------------

    texto = carregar_markdown(
        document_id
    )

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    chunks = executar_chunking(
        texto,
        test_id
    )

    print(
        f"   Chunks antes da validação: "
        f"{len(chunks)}"
    )

    # Teste Markdown pode produzir seções muito grandes.
    # Garantimos o limite do modelo de embeddings.
    chunks = (
        normalizar_chunks_limite_tokens(
            chunks
        )
    )

    print(
        f"   Chunks após validação: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    textos = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = gerar_embeddings(
        textos
    )

    if (
        len(embeddings)
        != len(chunks)
    ):

        raise RuntimeError(
            "Quantidade de embeddings diferente "
            "da quantidade de chunks."
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    resultado = []

    for indice, (
        chunk,
        embedding
    ) in enumerate(
        zip(
            chunks,
            embeddings
        ),
        start=1
    ):

        resultado.append({
            "chunk_id": (
                f"{document_id}_"
                f"test{test_id:02d}_"
                f"chunk{indice:04d}"
            ),

            "document_id":
                document_id,

            "document_name":
                f"{document_id}.pdf",

            "test_id":
                test_id,

            "strategy":
                config[
                    "strategy"
                ],

            "chunk_size":
                config[
                    "chunk_size"
                ],

            "chunk_overlap":
                config[
                    "chunk_overlap"
                ],

            "text":
                chunk[
                    "text"
                ],

            "embedding":
                embedding,

            "metadata":
                chunk[
                    "metadata"
                ],
        })

    with caminho.open(
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            resultado,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"   Dimensão: "
        f"{len(embeddings[0])}"
    )

    print(
        f"   OK -> {caminho}"
    )

    return "generated"


# ============================================================
# NORMALIZAR TODOS OS JSONs EXISTENTES
# ============================================================

def normalizar_todos_os_jsons():

    print("\n")
    print("=" * 70)
    print(
        "NORMALIZANDO SCHEMA DOS JSONs EXISTENTES"
    )
    print("=" * 70)

    total = 0

    for document_id in DOCUMENTOS:

        for test_id in range(
            1,
            11
        ):

            caminho = (
                PASTA_RESULTS
                / document_id
                / f"test_{test_id:02d}"
                / "chunks_embeddings.json"
            )

            if caminho.exists():

                normalizar_schema_arquivo(
                    caminho,
                    test_id
                )

                total += 1

    print(
        f"Arquivos verificados: {total}"
    )


# ============================================================
# VERIFICAR COBERTURA
# ============================================================

def verificar_cobertura():

    print("\n")
    print("=" * 70)
    print(
        "VERIFICAÇÃO DE COBERTURA"
    )
    print("=" * 70)

    faltantes = []

    for document_id in DOCUMENTOS:

        # Estes testes devem estar presentes
        # nos 12 documentos.
        for test_id in [
            1,
            2,
            3,
            9,
            10,
        ]:

            caminho = (
                PASTA_RESULTS
                / document_id
                / f"test_{test_id:02d}"
                / "chunks_embeddings.json"
            )

            if not caminho.exists():

                faltantes.append(
                    (
                        document_id,
                        test_id
                    )
                )

    # 4 a 8 precisam existir pelo menos
    # nos três documentos experimentais.
    documentos_experimentais = [
        "bioetica_e_ia",
        "escrita_academica_ia",
        "twitter_algoritmo",
    ]

    for document_id in (
        documentos_experimentais
    ):

        for test_id in range(
            4,
            9
        ):

            caminho = (
                PASTA_RESULTS
                / document_id
                / f"test_{test_id:02d}"
                / "chunks_embeddings.json"
            )

            if not caminho.exists():

                faltantes.append(
                    (
                        document_id,
                        test_id
                    )
                )

    if not faltantes:

        print(
            "\nCOBERTURA COMPLETA."
        )

        return True

    print(
        "\nArquivos faltantes:"
    )

    for (
        document_id,
        test_id
    ) in faltantes:

        print(
            f"- {document_id} "
            f"| Teste {test_id:02d}"
        )

    return False


# ============================================================
# MAIN
# ============================================================

def executar():

    print("\n")
    print("=" * 70)
    print(
        "ATIVIDADE 04 - ETAPA 5B"
    )
    print(
        "COMPLEMENTO DOS TESTES 01, 02 E 10"
    )
    print("=" * 70)

    print(
        f"\nModelo: "
        f"{EMBEDDING_MODEL}"
    )

    total = (
        len(DOCUMENTOS)
        * len(
            TESTES_COMPLEMENTARES
        )
    )

    print(
        f"Combinações verificadas: "
        f"{total}"
    )

    gerados = 0
    existentes = 0
    erros = []

    contador = 0

    for document_id in DOCUMENTOS:

        print("\n")
        print("#" * 70)
        print(
            f"DOCUMENTO: "
            f"{document_id}"
        )
        print("#" * 70)

        for test_id in (
            TESTES_COMPLEMENTARES
        ):

            contador += 1

            print(
                f"\nExperimento "
                f"{contador}/{total}"
            )

            print(
                f"Teste {test_id:02d} "
                f"- "
                f"{CONFIGURACOES[test_id]['strategy']}"
            )

            try:

                status = processar(
                    document_id,
                    test_id
                )

                if (
                    status
                    == "generated"
                ):

                    gerados += 1

                else:

                    existentes += 1

            except Exception as erro:

                print(
                    f"   ERRO: {erro}"
                )

                erros.append({
                    "document_id":
                        document_id,

                    "test_id":
                        test_id,

                    "error":
                        str(erro)
                })

    # --------------------------------------------------------
    # NORMALIZAR SCHEMA
    # --------------------------------------------------------

    normalizar_todos_os_jsons()

    # --------------------------------------------------------
    # COBERTURA
    # --------------------------------------------------------

    cobertura_ok = (
        verificar_cobertura()
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    resumo = {
        "embedding_model":
            EMBEDDING_MODEL,

        "tests_completed":
            TESTES_COMPLEMENTARES,

        "generated":
            gerados,

        "existing":
            existentes,

        "errors":
            erros,

        "coverage_ok":
            cobertura_ok,
    }

    caminho_resumo = (
        PASTA_RESULTS
        / "summary_cobertura.json"
    )

    with caminho_resumo.open(
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            resumo,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "RESUMO DA ETAPA 5B"
    )
    print("=" * 70)

    print(
        f"\nNovos resultados: "
        f"{gerados}"
    )

    print(
        f"Já existentes: "
        f"{existentes}"
    )

    print(
        f"Erros: "
        f"{len(erros)}"
    )

    print(
        f"Cobertura completa: "
        f"{cobertura_ok}"
    )

    print(
        f"\nResumo:\n"
        f"{caminho_resumo}"
    )

    print("\n")
    print("=" * 70)

    if (
        not erros
        and cobertura_ok
    ):

        print(
            "ETAPA 5B FINALIZADA COM SUCESSO"
        )

    else:

        print(
            "ETAPA 5B FINALIZADA COM PENDÊNCIAS"
        )

    print("=" * 70)


if __name__ == "__main__":
    executar()