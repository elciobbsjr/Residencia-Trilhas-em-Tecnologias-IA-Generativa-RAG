import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Pasta onde está este arquivo:
# projeto/aula_03/atividade_03.py
PASTA_ATUAL = Path(__file__).resolve().parent

# Raiz do projeto
ROOT = PASTA_ATUAL.parent

# Carrega o .env que está na raiz
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "openai/text-embedding-3-small"
)

if not OPENAI_API_KEY:
    raise ValueError(
        "A variável OPENAI_API_KEY não foi encontrada no arquivo .env."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

# Pasta dos arquivos Markdown da aula anterior
PASTA_MARKDOWN = ROOT / "aula_2"


# ============================================================
# 1. DISTÂNCIA EUCLIDIANA
# ============================================================

def distancia_euclidiana(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Calcula a distância euclidiana entre dois vetores.

    Fórmula:
    d(u, v) = sqrt(sum((u_i - v_i)^2))
    """

    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)

    if vec1.shape != vec2.shape:
        raise ValueError(
            "Os dois embeddings precisam possuir a mesma dimensão."
        )

    return float(np.linalg.norm(vec1 - vec2))


# ============================================================
# 2. SIMILARIDADE E DISTÂNCIA DE COSSENO
# ============================================================

def similaridade_cosseno(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Calcula a similaridade de cosseno entre dois vetores.

    Quanto mais próximo de 1, maior a similaridade.
    """

    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)

    if vec1.shape != vec2.shape:
        raise ValueError(
            "Os dois embeddings precisam possuir a mesma dimensão."
        )

    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return float(
        np.dot(vec1, vec2) / (norm_v1 * norm_v2)
    )


def distancia_cosseno(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Calcula a distância de cosseno.

    Distância = 1 - similaridade
    """

    return float(
        1.0 - similaridade_cosseno(vec1, vec2)
    )


# ============================================================
# FUNÇÕES PARA GERAR EMBEDDINGS
# ============================================================

def get_embedding(texto: str) -> np.ndarray:
    """
    Gera o embedding de um texto.
    """

    texto = texto.strip()

    if not texto:
        raise ValueError(
            "Não é possível gerar embedding de um texto vazio."
        )

    resposta = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texto
    )

    return np.array(
        resposta.data[0].embedding,
        dtype=np.float32
    )


def get_embeddings(textos: list[str]) -> list[np.ndarray]:
    """
    Gera embeddings para vários textos de uma vez.
    """

    textos = [texto.strip() for texto in textos if texto.strip()]

    if not textos:
        return []

    resposta = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=textos
    )

    return [
        np.array(item.embedding, dtype=np.float32)
        for item in resposta.data
    ]


# ============================================================
# PARTE 1 - TESTE COM VETORES NUMÉRICOS
# ============================================================

def teste_vetores_numericos():

    print("\n")
    print("=" * 70)
    print("PARTE 1 - TESTE COM VETORES NUMÉRICOS")
    print("=" * 70)

    embedding_a = np.array([1, 0, 0], dtype=np.float32)
    embedding_b = np.array([0, 1, 0], dtype=np.float32)
    embedding_c = np.array([1, 0, 0], dtype=np.float32)

    pares = [
        ("A", embedding_a, "B", embedding_b),
        ("A", embedding_a, "C", embedding_c),
        ("B", embedding_b, "C", embedding_c)
    ]

    resultados = []

    for nome1, vec1, nome2, vec2 in pares:

        resultados.append({
            "Par": f"{nome1} x {nome2}",
            "Distância Euclidiana":
                round(distancia_euclidiana(vec1, vec2), 4),
            "Similaridade Cosseno":
                round(similaridade_cosseno(vec1, vec2), 4),
            "Distância Cosseno":
                round(distancia_cosseno(vec1, vec2), 4)
        })

    df = pd.DataFrame(resultados)

    print(df.to_string(index=False))


# ============================================================
# PARTE 2 - EMBEDDINGS DOS TERMOS
# ============================================================

def teste_termos():

    print("\n")
    print("=" * 70)
    print("PARTE 2 - EMBEDDINGS DOS TERMOS")
    print("=" * 70)

    termos = [
        "gato",
        "felino",
        "cachorro",
        "carro",
        "caminhão",
        "moto",
        "banana",
        "maçã",
        "goiaba"
    ]

    print("\nGerando embeddings dos termos...")

    vetores = get_embeddings(termos)

    embeddings = {
        termo: vetor
        for termo, vetor in zip(termos, vetores)
    }

    print(
        f"Dimensão dos embeddings: "
        f"{len(vetores[0])}"
    )

    # Pares escolhidos para observar relações semânticas
    pares_teste = [
        ("gato", "felino"),
        ("gato", "cachorro"),
        ("gato", "carro"),

        ("carro", "caminhão"),
        ("carro", "moto"),
        ("caminhão", "moto"),

        ("banana", "maçã"),
        ("banana", "goiaba"),
        ("maçã", "goiaba")
    ]

    resultados = []

    for termo1, termo2 in pares_teste:

        vec1 = embeddings[termo1]
        vec2 = embeddings[termo2]

        resultados.append({
            "Termo 1": termo1,
            "Termo 2": termo2,
            "Distância Euclidiana":
                round(distancia_euclidiana(vec1, vec2), 4),
            "Similaridade Cosseno":
                round(similaridade_cosseno(vec1, vec2), 4),
            "Distância Cosseno":
                round(distancia_cosseno(vec1, vec2), 4)
        })

    df = pd.DataFrame(resultados)

    print("\nResultados:")
    print(df.to_string(index=False))


# ============================================================
# PARTE 3 - COMPARAÇÃO DE FRASES
# ============================================================

def teste_frases():

    print("\n")
    print("=" * 70)
    print("PARTE 3 - COMPARAÇÃO DE FRASES")
    print("=" * 70)

    frase_ancora = (
        "O cachorro correu no parque e brincou com a bola."
    )

    frases_comparacao = [
        (
            "Similar",
            "Um cão estava correndo no jardim "
            "e brincando com seu brinquedo."
        ),
        (
            "Relacionado",
            "O gato dormiu na almofada da sala "
            "durante toda a tarde."
        ),
        (
            "Diferente",
            "A taxa de juros do banco central "
            "subiu dois pontos percentuais."
        ),
        (
            "Oposto/Negação",
            "Nenhum animal esteve no parque e "
            "o cão permaneceu preso em casa."
        )
    ]

    print("\nGerando embedding da frase âncora...")

    vec_ancora = get_embedding(frase_ancora)

    textos = [
        frase
        for _, frase in frases_comparacao
    ]

    vecs_comparacao = get_embeddings(textos)

    resultados = []

    for (categoria, frase), vec in zip(
        frases_comparacao,
        vecs_comparacao
    ):

        resultados.append({
            "Categoria": categoria,
            "Texto": frase,
            "Distância Euclidiana":
                round(
                    distancia_euclidiana(
                        vec_ancora,
                        vec
                    ),
                    4
                ),
            "Similaridade Cosseno":
                round(
                    similaridade_cosseno(
                        vec_ancora,
                        vec
                    ),
                    4
                ),
            "Distância Cosseno":
                round(
                    distancia_cosseno(
                        vec_ancora,
                        vec
                    ),
                    4
                )
        })

    df = pd.DataFrame(resultados)

    print("\nFrase âncora:")
    print(frase_ancora)

    print("\nResultados:")
    print(df.to_string(index=False))


# ============================================================
# LEITURA DOS DOCUMENTOS MARKDOWN
# ============================================================

def carregar_documentos():
    """
    Lê todos os arquivos .md presentes na pasta aula_2.
    """

    if not PASTA_MARKDOWN.exists():
        raise FileNotFoundError(
            f"A pasta não foi encontrada: {PASTA_MARKDOWN}"
        )

    arquivos = list(PASTA_MARKDOWN.glob("*.md"))

    if not arquivos:
        raise FileNotFoundError(
            "Nenhum arquivo Markdown foi encontrado em aula_2."
        )

    documentos = []

    for arquivo in arquivos:

        texto = arquivo.read_text(
            encoding="utf-8"
        )

        documentos.append({
            "arquivo": arquivo.name,
            "texto": texto
        })

    return documentos


# ============================================================
# DIVISÃO POR LINHAS
# ============================================================

def dividir_por_linhas(documentos):

    trechos = []

    for documento in documentos:

        linhas = documento["texto"].splitlines()

        for numero, linha in enumerate(
            linhas,
            start=1
        ):

            linha = linha.strip()

            # Ignora linhas vazias
            if not linha:
                continue

            trechos.append({
                "arquivo": documento["arquivo"],
                "local": f"Linha {numero}",
                "texto": linha
            })

    return trechos


# ============================================================
# DIVISÃO POR PARÁGRAFOS
# ============================================================

def dividir_por_paragrafos(documentos):

    trechos = []

    for documento in documentos:

        paragrafos = re.split(
            r"\n\s*\n",
            documento["texto"]
        )

        numero_paragrafo = 0

        for paragrafo in paragrafos:

            paragrafo = paragrafo.strip()

            if not paragrafo:
                continue

            numero_paragrafo += 1

            trechos.append({
                "arquivo": documento["arquivo"],
                "local":
                    f"Parágrafo {numero_paragrafo}",
                "texto": paragrafo
            })

    return trechos


# ============================================================
# DIVISÃO POR CAPÍTULOS
# ============================================================

def dividir_por_capitulos(documentos):
    """
    Divide os documentos em capítulos usando títulos de nível 2 (##).
    Subtítulos menores permanecem dentro do capítulo.

    Capítulos muito pequenos ou contendo apenas um título
    são ignorados.
    """

    trechos = []

    for documento in documentos:

        linhas = documento["texto"].splitlines()

        capitulos = []

        titulo_atual = "Introdução"
        conteudo_atual = []

        for linha in linhas:

            # Considera ## como início de um capítulo
            if re.match(r"^##\s+", linha):

                # Salva o capítulo anterior
                texto_capitulo = "\n".join(
                    conteudo_atual
                ).strip()

                # Ignora capítulos vazios ou muito pequenos
                if len(texto_capitulo) >= 80:

                    capitulos.append({
                        "titulo": titulo_atual,
                        "texto": texto_capitulo
                    })

                # Inicia um novo capítulo
                titulo_atual = re.sub(
                    r"^##\s+",
                    "",
                    linha
                ).strip()

                conteudo_atual = [linha]

            else:
                conteudo_atual.append(linha)

        # Salva o último capítulo
        texto_capitulo = "\n".join(
            conteudo_atual
        ).strip()

        if len(texto_capitulo) >= 80:

            capitulos.append({
                "titulo": titulo_atual,
                "texto": texto_capitulo
            })

        # Adiciona capítulos à lista geral
        for numero, capitulo in enumerate(
            capitulos,
            start=1
        ):

            trechos.append({
                "arquivo": documento["arquivo"],
                "local": (
                    f"Capítulo {numero} - "
                    f"{capitulo['titulo']}"
                ),
                "texto": capitulo["texto"]
            })

    return trechos
# ============================================================
# GERAR EMBEDDINGS DOS TRECHOS
# ============================================================

def indexar_trechos(trechos):
    """
    Gera o embedding de cada trecho.
    """

    textos = [
        trecho["texto"]
        for trecho in trechos
    ]

    print(
        f"Gerando embeddings de "
        f"{len(textos)} trechos..."
    )

    embeddings = get_embeddings(textos)

    for trecho, embedding in zip(
        trechos,
        embeddings
    ):
        trecho["embedding"] = embedding

    return trechos


# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def busca_semantica(
    query,
    trechos,
    top_k=3
):
    """
    Compara o embedding da query com os embeddings
    dos trechos utilizando similaridade de cosseno.
    """

    embedding_query = get_embedding(query)

    resultados = []

    for trecho in trechos:

        score = similaridade_cosseno(
            embedding_query,
            trecho["embedding"]
        )

        resultados.append({
            "arquivo": trecho["arquivo"],
            "local": trecho["local"],
            "texto": trecho["texto"],
            "score": score
        })

    # Quanto maior a similaridade, melhor
    resultados.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return resultados[:top_k]


# ============================================================
# EXIBIÇÃO DO TOP 3
# ============================================================

def mostrar_resultados(
    titulo,
    query,
    resultados
):

    print("\n")
    print("=" * 70)
    print(titulo)
    print("=" * 70)

    print(f"\nQUERY:")
    print(query)

    for posicao, resultado in enumerate(
        resultados,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"TOP {posicao} "
            f"| Similaridade: "
            f"{resultado['score']:.4f}"
        )

        print(
            f"Arquivo: "
            f"{resultado['arquivo']}"
        )

        print(
            f"Local: "
            f"{resultado['local']}"
        )

        print("\nTrecho:")

        # Evita imprimir capítulos gigantes no terminal
        texto = resultado["texto"]

        if len(texto) > 1000:
            texto = texto[:1000] + "..."

        print(texto)


# ============================================================
# PARTE 4 - BUSCA SEMÂNTICA NOS MARKDOWNS
# ============================================================

def executar_busca_semantica():

    print("\n")
    print("=" * 70)
    print("PARTE 4 - BUSCA SEMÂNTICA MANUAL")
    print("=" * 70)

    # ========================================================
    # CARREGAR DOCUMENTOS
    # ========================================================

    documentos = carregar_documentos()

    print(
        f"\nForam encontrados "
        f"{len(documentos)} documentos:"
    )

    for documento in documentos:
        print(f"- {documento['arquivo']}")

    # ========================================================
    # QUERIES
    # ========================================================

    queries = [
        "O que é autonomia e opacidade algorítmica?",
        "O que é o diário de bordo da IA?"
    ]

    # ========================================================
    # PREPARAR OS TRECHOS UMA ÚNICA VEZ
    # ========================================================

    print("\nPreparando busca por LINHAS...")

    trechos_linhas = dividir_por_linhas(
        documentos
    )

    trechos_linhas = indexar_trechos(
        trechos_linhas
    )

    print("\nPreparando busca por PARÁGRAFOS...")

    trechos_paragrafos = dividir_por_paragrafos(
        documentos
    )

    trechos_paragrafos = indexar_trechos(
        trechos_paragrafos
    )

    print("\nPreparando busca por CAPÍTULOS...")

    trechos_capitulos = dividir_por_capitulos(
        documentos
    )

    trechos_capitulos = indexar_trechos(
        trechos_capitulos
    )

    # ========================================================
    # EXECUTAR TODAS AS QUERIES
    # ========================================================

    for numero_query, query in enumerate(
        queries,
        start=1
    ):

        print("\n")
        print("#" * 70)
        print(
            f"QUERY {numero_query} DE {len(queries)}"
        )
        print("#" * 70)

        print(f"\nPergunta:")
        print(query)

        # ====================================================
        # BUSCA POR LINHAS
        # ====================================================

        resultados_linhas = busca_semantica(
            query,
            trechos_linhas,
            top_k=3
        )

        mostrar_resultados(
            "BUSCA SEMÂNTICA - LINHAS",
            query,
            resultados_linhas
        )

        # ====================================================
        # BUSCA POR PARÁGRAFOS
        # ====================================================

        resultados_paragrafos = busca_semantica(
            query,
            trechos_paragrafos,
            top_k=3
        )

        mostrar_resultados(
            "BUSCA SEMÂNTICA - PARÁGRAFOS",
            query,
            resultados_paragrafos
        )

        # ====================================================
        # BUSCA POR CAPÍTULOS
        # ====================================================

        resultados_capitulos = busca_semantica(
            query,
            trechos_capitulos,
            top_k=3
        )

        mostrar_resultados(
            "BUSCA SEMÂNTICA - CAPÍTULOS",
            query,
            resultados_capitulos
        )

# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("\nATIVIDADE 03 - EMBEDDINGS E BUSCA SEMÂNTICA")

    teste_vetores_numericos()

    teste_termos()

    teste_frases()

    executar_busca_semantica()

    print("\n")
    print("=" * 70)
    print("ATIVIDADE FINALIZADA")
    print("=" * 70)


if __name__ == "__main__":
    main()