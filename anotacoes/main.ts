import { readFile } from "fs/promises";

type Artigo = {
    nome: string
    url: string
    ano: number
    revisado: boolean
}

async function main() {
    const file = await readFile('C:/MinhasCoisas/Programacao/Workspace/TCC-II/anotacoes/assets/data/artigos-que-citam-o-artigo-do-hassan.json')
    const array = JSON.parse(file) as Artigo[];
    console.log(array.filter(a => a.revisado))
}

main()