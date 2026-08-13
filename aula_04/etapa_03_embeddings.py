import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CAMINHOS
# ============================================================

PASTA_ATUAL = Path(__file__).resolve().parent
ROOT = PASTA_ATUAL.parent

PASTA_RESULTS = PASTA_ATUAL / "results"


# ============================================================
# CARREGAR .ENV
# ============================================================

load_dotenv(ROOT / ".env")


# ============================================================
# CONFIGURAÇÃO DA API
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "openai/text-embedding-3-small"
)


if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY não encontrada no arquivo .env."
    )


# ============================================================
# CLIENTE OPENROUTER
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)


# ============================================================
# DOCUMENTOS USADOS NOS 10 EXPERIMENTOS
# ============================================================

DOCUMENTOS_TESTE = [
    "bioetica_e_ia",
    "escrita_academica_ia",
    "twitter_algoritmo",
]


# ============================================================
# CONFIGURAÇÃO
# ============================================================

TOTAL_TESTES = 10

# Quantos textos serão enviados por requisição.
BATCH_SIZE = 64

# Pequena pausa entre batches.
PAUSA_ENTRE_BATCHES = 0.2


# ============================================================
# GERAR EMBEDDINGS EM LOTE
# ============================================================

def gerar_embeddings_lote(
    textos,
    batch_size=BATCH_SIZE
):
    """
    Gera embeddings para vários textos.

    Os textos são enviados em pequenos lotes para evitar
    requisições muito grandes.
    """

    todos_embeddings = []

    total = len(textos)

    for inicio in range(
        0,
        total,
        batch_size
    ):

        fim = min(
            inicio + batch_size,
            total
        )

        lote = textos[
            inicio:fim
        ]

        print(
            f"      Batch "
            f"{inicio + 1}-{fim}/{total}"
        )

        resposta = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=lote
        )

        embeddings = [
            item.embedding
            for item in resposta.data
        ]

        todos_embeddings.extend(
            embeddings
        )

        time.sleep(
            PAUSA_ENTRE_BATCHES
        )

    return todos_embeddings


# ============================================================
# CARREGAR CHUNKS PRELIMINARES
# ============================================================

def carregar_chunks(
    document_id,
    test_id
):
    """
    Carrega o arquivo chunks_preliminares.json produzido
    na Etapa 2.
    """

    caminho = (
        PASTA_RESULTS
        / document_id
        / f"test_{test_id:02d}"
        / "chunks_preliminares.json"
    )

    if not caminho.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{caminho}"
        )

    with caminho.open(
        "r",
        encoding="utf-8"
    ) as arquivo:

        return json.load(
            arquivo
        )


# ============================================================
# SALVAR JSON COM EMBEDDINGS
# ============================================================

def salvar_chunks_embeddings(
    document_id,
    test_id,
    chunks
):
    """
    Salva o resultado final do experimento.
    """

    pasta_teste = (
        PASTA_RESULTS
        / document_id
        / f"test_{test_id:02d}"
    )

    caminho = (
        pasta_teste
        / "chunks_embeddings.json"
    )

    with caminho.open(
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            chunks,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    return caminho


# ============================================================
# PROCESSAR UM EXPERIMENTO
# ============================================================

def processar_experimento(
    document_id,
    test_id
):
    """
    Carrega os chunks de um teste, gera os embeddings
    e salva chunks_embeddings.json.
    """

    print("\n" + "-" * 70)

    print(
        f"Documento: {document_id}"
    )

    print(
        f"Teste: {test_id:02d}"
    )

    # --------------------------------------------------------
    # CAMINHO FINAL
    # --------------------------------------------------------

    caminho_final = (
        PASTA_RESULTS
        / document_id
        / f"test_{test_id:02d}"
        / "chunks_embeddings.json"
    )

    # --------------------------------------------------------
    # EVITAR PROCESSAR NOVAMENTE
    # --------------------------------------------------------

    if (
        caminho_final.exists()
        and caminho_final.stat().st_size > 0
    ):

        print(
            "   chunks_embeddings.json já existe."
        )

        print(
            "   Teste ignorado."
        )

        with caminho_final.open(
            "r",
            encoding="utf-8"
        ) as arquivo:

            chunks_existentes = json.load(
                arquivo
            )

        if chunks_existentes:

            embedding = chunks_existentes[
                0
            ].get(
                "embedding"
            )

            dimensao = (
                len(embedding)
                if embedding
                else None
            )

        else:
            dimensao = None

        return {
            "document_id":
                document_id,

            "test_id":
                test_id,

            "num_chunks":
                len(
                    chunks_existentes
                ),

            "embedding_dimension":
                dimensao,

            "status":
                "existing"
        }

    # --------------------------------------------------------
    # CARREGAR CHUNKS
    # --------------------------------------------------------

    chunks = carregar_chunks(
        document_id,
        test_id
    )

    print(
        f"   Chunks encontrados: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # PEGAR TEXTOS
    # --------------------------------------------------------

    textos = [
        chunk["text"].strip()
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # VALIDAR
    # --------------------------------------------------------

    if not textos:

        raise ValueError(
            "Nenhum chunk encontrado."
        )

    if any(
        not texto
        for texto in textos
    ):

        raise ValueError(
            "Foi encontrado um chunk vazio."
        )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    print(
        "   Gerando embeddings..."
    )

    embeddings = gerar_embeddings_lote(
        textos
    )

    # --------------------------------------------------------
    # VALIDAR QUANTIDADE
    # --------------------------------------------------------

    if (
        len(embeddings)
        != len(chunks)
    ):

        raise RuntimeError(
            "A quantidade de embeddings gerados "
            "não corresponde à quantidade de chunks."
        )

    # --------------------------------------------------------
    # ADICIONAR EMBEDDING
    # --------------------------------------------------------

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        chunk[
            "embedding"
        ] = embedding

    # --------------------------------------------------------
    # DIMENSÃO
    # --------------------------------------------------------

    embedding_dimension = len(
        embeddings[0]
    )

    print(
        f"   Dimensão: "
        f"{embedding_dimension}"
    )

    # --------------------------------------------------------
    # SALVAR
    # --------------------------------------------------------

    caminho = salvar_chunks_embeddings(
        document_id,
        test_id,
        chunks
    )

    print(
        f"   OK -> {caminho}"
    )

    return {
        "document_id":
            document_id,

        "test_id":
            test_id,

        "num_chunks":
            len(chunks),

        "embedding_dimension":
            embedding_dimension,

        "status":
            "generated"
    }


# ============================================================
# EXECUTAR TODOS OS EXPERIMENTOS
# ============================================================

def executar_embeddings():

    print("\n")
    print("=" * 70)
    print(
        "ATIVIDADE 04 - ETAPA 3"
    )
    print(
        "GERAÇÃO DOS EMBEDDINGS"
    )
    print("=" * 70)

    print(
        f"\nModelo: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Documentos: "
        f"{len(DOCUMENTOS_TESTE)}"
    )

    print(
        f"Testes por documento: "
        f"{TOTAL_TESTES}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    resultados = []

    erros = []

    total_experimentos = (
        len(DOCUMENTOS_TESTE)
        * TOTAL_TESTES
    )

    contador = 0

    # ========================================================
    # DOCUMENTOS
    # ========================================================

    for document_id in DOCUMENTOS_TESTE:

        print("\n")
        print("#" * 70)
        print(
            f"DOCUMENTO: {document_id}"
        )
        print("#" * 70)

        # ====================================================
        # TESTES
        # ====================================================

        for test_id in range(
            1,
            TOTAL_TESTES + 1
        ):

            contador += 1

            print(
                f"\nExperimento "
                f"{contador}/{total_experimentos}"
            )

            try:

                resultado = (
                    processar_experimento(
                        document_id,
                        test_id
                    )
                )

                resultados.append(
                    resultado
                )

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

    # ========================================================
    # RESUMO
    # ========================================================

    summary = {
        "embedding_model":
            EMBEDDING_MODEL,

        "documents":
            DOCUMENTOS_TESTE,

        "num_documents":
            len(
                DOCUMENTOS_TESTE
            ),

        "num_experiments_expected":
            total_experimentos,

        "num_experiments_success":
            len(resultados),

        "num_errors":
            len(erros),

        "results":
            resultados,

        "errors":
            erros
    }

    # ========================================================
    # SALVAR SUMMARY
    # ========================================================

    caminho_summary = (
        PASTA_RESULTS
        / "summary_embeddings.json"
    )

    with caminho_summary.open(
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            summary,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # MOSTRAR RESUMO
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "RESUMO DOS EMBEDDINGS"
    )
    print("=" * 70)

    print(
        f"\nExperimentos esperados: "
        f"{total_experimentos}"
    )

    print(
        f"Experimentos concluídos: "
        f"{len(resultados)}"
    )

    print(
        f"Erros: "
        f"{len(erros)}"
    )

    if resultados:

        dimensoes = {
            resultado[
                "embedding_dimension"
            ]
            for resultado in resultados
            if resultado[
                "embedding_dimension"
            ]
            is not None
        }

        print(
            f"Dimensões encontradas: "
            f"{sorted(dimensoes)}"
        )

    # ========================================================
    # ERROS
    # ========================================================

    if erros:

        print(
            "\nEXPERIMENTOS COM ERRO:"
        )

        for erro in erros:

            print(
                f"- "
                f"{erro['document_id']} "
                f"| Teste "
                f"{erro['test_id']:02d}: "
                f"{erro['error']}"
            )

    print(
        f"\nResumo salvo em:\n"
        f"{caminho_summary}"
    )

    print("\n")
    print("=" * 70)

    if erros:

        print(
            "ETAPA 3 FINALIZADA COM ERROS"
        )

    else:

        print(
            "ETAPA 3 FINALIZADA COM SUCESSO"
        )

    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    executar_embeddings()