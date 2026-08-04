import tf, { model } from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    // ADAM (Adaptive Moment Estimation) - Treinador moderno para redes neurais (ajusta os pesos de forma eficiente e inteligente com historico de erros e acertos)
    
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // categoria premium sempre será [1, 0, 0]

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss)
    // Exemplo classico: classificação de imagens, recomendação, categorização de usuário
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    model.compile({
        optimizer: 'adam', 
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamento do modelo
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0, // desabilita o log interno e usa só o callback
            epochs: 100, // passa 100 vezes por toda a base de dados
            shuffle: true, // embaralha os dados
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(`Epoch: ${epoch} | loss = ${log.loss} `)
            }
        }
    )

    return model;
}

async function predict(model, pessoa) {
    // transformar o array js para tensor
    const tfInput = tf.tensor2d(pessoa)

    // faz predição (output de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}

const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Gera um one-hot aleatório de tamanho `size`
function oneHotAleatorio(size) {
    const vetor = new Array(size).fill(0)
    vetor[Math.floor(Math.random() * size)] = 1
    return vetor
}

// Aumenta a base em 50 pessoas com dados totalmente aleatórios
for (let i = 0; i < 50; i++) {
    const idade = Math.random()
    const cor = oneHotAleatorio(3)
    const localizacao = oneHotAleatorio(3)
    tensorPessoasNormalizado.push([idade, ...cor, ...localizacao])
    tensorLabels.push(oneHotAleatorio(3))
}

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dados melhor
const models = await trainModel(inputXs, outputYs);

const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: 'Curitiba' }

const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        0,    // cor azul
        1,    // cor vermelho
        0,    // cor verde
        1,    // localização São Paulo
        0,    // localização Rio
        0     // localização Curitiba
    ]
]

const predictions = await predict(models, pessoaTensorNormalizado)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(results)