/**
 * Генетический алгоритм — Вариант 10
 * Задача: max f(x) = x^2 + 20x − 34,  x ∈ {8, 9, 10, 11, 12}  (дискретный интервал)
 *
 * Операторы (вариант 10):
 *   I(A)      — бинарное кодирование хромосом
 *   II(A,B)   — инициализация: «одеяло» (A) или «дробовик» (B)
 *   III(B,E)  — селекция: рулетка (B) или инбридинг (E)
 *   IV(A,B,E) — кроссинговер: одноточечный (A), двухточечный (B),
 *               упорядочивающий одноточечный (E)
 *   V(A,F)    — мутация: простая (A), инверсия (F)
 *   VI(B)     — элитный отбор
 *   VII(A)    — микроэволюция (генерационная замена)
 *
 * Вывод:
 *   ga_results.csv    — статистика по каждому поколению
 *   ga_population.csv — все особи каждого поколения (x, f(x))
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>
#include <locale>
#include <numeric>
#include <sstream>

 //                                                                              
 //  Параметры задачи
 //                                                                              
 // Дискретное множество допустимых значений: {8, 9, 10, 11, 12}
const int X_VALUES[] = { 8, 9, 10, 11, 12 };
const int NUM_X = 5;          // количество допустимых значений
const int NUM_BITS = 3;          // 2^3=8 комбинаций, используем индексы 0..4

//                                                                              
//  Типы
//                                                                              
using Chromosome = std::vector<int>;   // биты
using Population = std::vector<Chromosome>;

//                                                                              
//  Глобальный генератор псевдослучайных чисел
//                                                                              
std::mt19937 rng;

//                                                                              
//  Кодирование / декодирование
//                                                                              

/// Хромосома (3 бита) → целый индекс, зажатый в [0, NUM_X-1]
int decodeIndex(const Chromosome& ch) {
    int value = 0;
    for (int i = 0; i < NUM_BITS; ++i)
        value = (value << 1) | ch[i];
    // Зажать: значения 5,6,7 → 4 (x=12)
    return std::min(value, NUM_X - 1);
}

/// Хромосома → допустимое x из таблицы
int decodeX(const Chromosome& ch) {
    return X_VALUES[decodeIndex(ch)];
}

/// Индекс (0..4) → хромосома
Chromosome encodeIndex(int idx) {
    idx = std::max(0, std::min(idx, NUM_X - 1));
    Chromosome ch(NUM_BITS);
    for (int i = NUM_BITS - 1; i >= 0; --i) {
        ch[i] = idx & 1;
        idx >>= 1;
    }
    return ch;
}

//                                                                              
//  Целевая функция
//                                                                              
double fitnessFunc(int x) {
    return static_cast<double>(x * x + 20 * x - 34);
}

double fitness(const Chromosome& ch) {
    return fitnessFunc(decodeX(ch));
}

//                                                                              
//  II. Инициализация начальной популяции
//                                                                              

/// «Одеяло» (A): равномерно покрыть все допустимые значения.
/// При popSize > NUM_X — циклически повторяем значения.
Population blanketInit(int popSize) {
    Population pop;
    for (int i = 0; i < popSize; ++i)
        pop.push_back(encodeIndex(i % NUM_X));
    return pop;
}

/// «Дробовик» (B): случайный индекс из [0, NUM_X-1]
Population shotgunInit(int popSize) {
    std::uniform_int_distribution<int> dist(0, NUM_X - 1);
    Population pop;
    for (int i = 0; i < popSize; ++i)
        pop.push_back(encodeIndex(dist(rng)));
    return pop;
}

//                                                                              
//  III. Методы селекции родителей
//                                                                              

/// Сдвинуть приспособленности так, чтобы минимум ≥ 1
std::vector<double> shiftedFitness(const std::vector<double>& fits) {
    double minF = *std::min_element(fits.begin(), fits.end());
    double shift = (minF < 1.0) ? (1.0 - minF) : 0.0;
    std::vector<double> sf(fits.size());
    for (size_t i = 0; i < fits.size(); ++i)
        sf[i] = fits[i] + shift;
    return sf;
}

/// Рулетка (B): выбрать индекс пропорционально приспособленности
int rouletteSelect(const std::vector<double>& fits) {
    auto sf = shiftedFitness(fits);
    double total = 0.0;
    for (double f : sf) total += f;
    std::uniform_real_distribution<double> dist(0.0, total);
    double r = dist(rng), cumsum = 0.0;
    for (size_t i = 0; i < sf.size(); ++i) {
        cumsum += sf[i];
        if (r <= cumsum) return static_cast<int>(i);
    }
    return static_cast<int>(sf.size()) - 1;
}

/// Расстояние Хэмминга
int hammingDist(const Chromosome& a, const Chromosome& b) {
    int d = 0;
    for (int i = 0; i < NUM_BITS; ++i) d += (a[i] != b[i]);
    return d;
}

/// Инбридинг (E): первый — рулетка, второй — наиболее похожий
std::pair<int, int> inbreedingSelect(const Population& pop,
    const std::vector<double>& fits) {
    int p1 = rouletteSelect(fits);
    int p2 = -1, minD = NUM_BITS + 1;
    for (int i = 0; i < static_cast<int>(pop.size()); ++i) {
        if (i == p1) continue;
        int d = hammingDist(pop[p1], pop[i]);
        if (d < minD) { minD = d; p2 = i; }
    }
    if (p2 == -1) p2 = (p1 + 1) % pop.size();
    return { p1, p2 };
}

//                                                                              
//  IV. Операторы кроссинговера
//                                                                              
using CrossoverPair = std::pair<Chromosome, Chromosome>;

/// Одноточечный (A)
CrossoverPair singlePointCrossover(const Chromosome& p1, const Chromosome& p2) {
    // При NUM_BITS=3 возможные точки: 1 или 2
    std::uniform_int_distribution<int> dist(1, NUM_BITS - 1);
    int pt = dist(rng);
    Chromosome c1 = p1, c2 = p2;
    for (int i = pt; i < NUM_BITS; ++i) std::swap(c1[i], c2[i]);
    return { c1, c2 };
}

/// Двухточечный (B)
/// При NUM_BITS=3 единственная «внутренняя» пара точек: (1,1) и (1,2) и (2,2)
CrossoverPair twoPointCrossover(const Chromosome& p1, const Chromosome& p2) {
    std::uniform_int_distribution<int> dist(1, NUM_BITS - 1);
    int a = dist(rng), b = dist(rng);
    if (a > b) std::swap(a, b);
    Chromosome c1 = p1, c2 = p2;
    for (int i = a; i <= b; ++i) std::swap(c1[i], c2[i]);
    return { c1, c2 };
}

/// Упорядочивающий одноточечный (E)
/// Биты правых частей обоих родителей объединяются и сортируются:
/// c1 получает биты в порядке возрастания (→ меньший индекс → меньший x),
/// c2 — в порядке убывания (→ больший x).
CrossoverPair orderingSinglePointCrossover(const Chromosome& p1, const Chromosome& p2) {
    std::uniform_int_distribution<int> dist(1, NUM_BITS - 1);
    int pt = dist(rng);
    int rightLen = NUM_BITS - pt;

    // Собрать биты правых частей
    std::vector<int> bits;
    for (int i = pt; i < NUM_BITS; ++i) bits.push_back(p1[i]);
    for (int i = pt; i < NUM_BITS; ++i) bits.push_back(p2[i]);

    std::vector<int> asc = bits, desc = bits;
    std::sort(asc.begin(), asc.end());
    std::sort(desc.begin(), desc.end(), std::greater<int>());

    Chromosome c1(p1.begin(), p1.begin() + pt);
    Chromosome c2(p2.begin(), p2.begin() + pt);
    for (int i = 0; i < rightLen; ++i) { c1.push_back(asc[i]); }
    for (int i = 0; i < rightLen; ++i) { c2.push_back(desc[i]); }

    // Убедиться в корректной длине
    c1.resize(NUM_BITS, 0);
    c2.resize(NUM_BITS, 0);
    return { c1, c2 };
}

//                                                                              
//  V. Операторы мутации
//                                                                              

/// Простая мутация (A): инвертировать один случайный бит
Chromosome simpleMutation(const Chromosome& ch) {
    std::uniform_int_distribution<int> dist(0, NUM_BITS - 1);
    Chromosome result = ch;
    result[dist(rng)] ^= 1;
    return result;
}

/// Инверсия (F): развернуть случайный подотрезок
Chromosome inversionMutation(const Chromosome& ch) {
    std::uniform_int_distribution<int> dist(0, NUM_BITS - 1);
    int a = dist(rng), b = dist(rng);
    if (a > b) std::swap(a, b);
    Chromosome result = ch;
    std::reverse(result.begin() + a, result.begin() + b + 1);
    return result;
}

//                                                                              
//  VI. Элитный отбор — перенести eliteCount лучших в следующее поколение
//                                                                              
Population eliteSelection(const Population& pop,
    const std::vector<double>& fits,
    int eliteCount) {
    std::vector<int> idx(pop.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&](int a, int b) { return fits[a] > fits[b]; });
    Population elite;
    for (int i = 0; i < eliteCount && i < static_cast<int>(pop.size()); ++i)
        elite.push_back(pop[idx[i]]);
    return elite;
}

//                                                                              
//  Вывод таблицы популяции в консоль
//                                                                              

// Подсчёт визуальной ширины UTF-8 строки:
// каждый ASCII-байт = 1 колонка, каждый многобайтовый символ (кириллица) = 1 колонка
// но занимает 2 байта → нужно вычесть лишние байты из setw-значения
int utf8VisualWidth(const std::string& s) {
    int w = 0;
    for (unsigned char c : s) {
        // Считаем только ведущие байты (не continuation bytes 10xxxxxx)
        if ((c & 0xC0) != 0x80) ++w;
    }
    return w;
}

// Вывести строку с дополнением пробелами до visualWidth колонок
void padPrint(const std::string& s, int visualWidth) {
    std::cout << s;
    int pad = visualWidth - utf8VisualWidth(s);
    if (pad > 0) std::cout << std::string(pad, ' ');
}

void printPopulationTable(const std::string& title, const Population& pop) {
    const std::string sep(68, '-');
    std::cout << "\n" << title << "\n";
    std::cout << sep << "\n";
    padPrint("Номер", 12);
    padPrint("x", 10);
    padPrint("Целевая функция f(x)", 24);
    std::cout << "Хромосома\n";
    std::cout << sep << "\n";
    for (int i = 0; i < static_cast<int>(pop.size()); ++i) {
        int x = decodeX(pop[i]);
        int f = static_cast<int>(fitnessFunc(x));
        std::string chStr;
        for (int b : pop[i]) chStr += std::to_string(b);
        padPrint("Особь " + std::to_string(i), 12);
        padPrint(std::to_string(x), 10);
        padPrint(std::to_string(f), 24);
        std::cout << chStr << "\n";
    }
    std::cout << sep << "\n";
}

//                                                                              
//  Вспомогательные функции ввода
//                                                                              
template<typename T>
T promptValue(const std::string& prompt, T defaultVal) {
    std::cout << prompt << " [" << defaultVal << "]: ";
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return defaultVal;
    std::istringstream ss(line);
    T val; ss >> val;
    return val;
}

int promptChoice(const std::string& prompt,
    const std::vector<std::string>& options,
    int defaultIdx) {
    std::cout << "\n" << prompt << "\n";
    for (size_t i = 0; i < options.size(); ++i)
        std::cout << "  " << (i + 1) << " — " << options[i] << "\n";
    int val = promptValue<int>("Выбор", defaultIdx);
    if (val < 1 || val > static_cast<int>(options.size())) val = defaultIdx;
    return val;
}

//                                                                              
//  MAIN
//                                                                              
int main() {
    setlocale(LC_ALL, "ru");

    std::random_device rd;
    rng.seed(rd());

    std::cout << "   Генетический алгоритм — Вариант 10\n";
    std::cout << "   max f(x) = x^2 + 20x - 34,  x принадлежит {8, 9, 10, 11, 12}\n";
    std::cout << "   Кодирование: 3 бита (индекс 0..4  x = 8..12)\n";

    // Вывести таблицу допустимых значений для наглядности
    std::cout << "  Таблица кодирования:\n";
    std::cout << "                         \n";
    std::cout << "    Биты    x     f(x)   \n";
    std::cout << "                         \n";
    for (int i = 0; i < NUM_X; ++i) {
        Chromosome ch = encodeIndex(i);
        std::cout << "    ";
        for (int b : ch) std::cout << b;
        std::cout << "     " << X_VALUES[i] << "    "
            << std::setw(6) << fitnessFunc(X_VALUES[i]) << "  \n";
    }
    std::cout << "    101     12      350    \n";
    std::cout << "                         \n\n";

    //    Параметры                                                              
    int    popSize = promptValue<int>("Размер популяции", 10);
    int    numGens = promptValue<int>("Число генераций", 50);
    double pCross = promptValue<double>("Вероятность кроссинговера", 0.70);
    double pMut = promptValue<double>("Вероятность мутации", 0.20);
    int    eliteCount = promptValue<int>("Число элитных особей", 2);

    if (popSize < 2)  popSize = 2;
    if (numGens < 50) {
        std::cout << "[!] Число генераций < 50; установлено 50.\n";
        numGens = 50;
    }
    if (eliteCount >= popSize) eliteCount = std::max(1, popSize / 5);

    int initChoice = promptChoice(
        "II. Стратегия начальной популяции",
        { "Одеяло — покрыть все значения {8..12} равномерно",
         "Дробовик — случайный выбор из {8..12}" },
        1);

    int selChoice = promptChoice(
        "III. Метод селекции родителей",
        { "Рулетка (пропорционально f(x))",
         "Инбридинг (предпочтение похожих по Хэммингу)" },
        1);

    int crossChoice = promptChoice(
        "IV. Оператор кроссинговера",
        { "Одноточечный",
         "Двухточечный",
         "Упорядочивающий одноточечный" },
        1);

    int mutChoice = promptChoice(
        "V. Оператор мутации",
        { "Простая (инверсия одного бита)",
         "Инверсия подотрезка" },
        1);

    //    Инициализация                                                          
    Population pop = (initChoice == 1)
        ? blanketInit(popSize)
        : shotgunInit(popSize);

    //    Вывод начальной популяции                                             
    printPopulationTable("Начальная популяция:", pop);

    //    CSV файлы                                                              
    std::ofstream csvStats("ga_results.csv");
    csvStats << "generation;best_x;best_f;avg_f;worst_f;std_f;best_chromosome\n";

    std::ofstream csvPop("ga_population.csv");
    csvPop << "generation;individual;x;f_x;chromosome\n";

    csvStats.imbue(std::locale("ru"));
    //    Глобальный оптимум                                                     
    Chromosome globalBestCh;
    double     globalBestFit = -1e18;
    int        globalBestX = 8;

    std::uniform_real_distribution<double> prob(0.0, 1.0);

    std::cout << "\n  Старт эволюции...\n\n";

    //    Главный цикл (VII — микроэволюция)                                    
    for (int gen = 0; gen < numGens; ++gen) {

        // Приспособленность
        std::vector<double> fits(popSize);
        for (int i = 0; i < popSize; ++i)
            fits[i] = fitness(pop[i]);

        // Статистика
        int bestIdx = static_cast<int>(std::max_element(fits.begin(), fits.end()) - fits.begin());
        int worstIdx = static_cast<int>(std::min_element(fits.begin(), fits.end()) - fits.begin());
        double bestFit = fits[bestIdx];
        double worstFit = fits[worstIdx];
        int    bestX = decodeX(pop[bestIdx]);

        double avgFit = 0.0;
        for (double f : fits) avgFit += f;
        avgFit /= popSize;

        double variance = 0.0;
        for (double f : fits) variance += (f - avgFit) * (f - avgFit);
        double stdFit = std::sqrt(variance / popSize);

        if (bestFit > globalBestFit) {
            globalBestFit = bestFit;
            globalBestX = bestX;
            globalBestCh = pop[bestIdx];
        }

        // Лучшая хромосома строкой
        std::string bestChStr;
        for (int b : pop[bestIdx]) bestChStr += std::to_string(b);

        // Запись статистики
        csvStats << gen << ";"
            << bestX << ";"
            << std::fixed << std::setprecision(2)
            << bestFit << ";"
            << avgFit << ";"
            << worstFit << ";"
            << stdFit << ";"
            << bestChStr << "\n";

        // Запись всех особей
        for (int i = 0; i < popSize; ++i) {
            std::string chStr;
            for (int b : pop[i]) chStr += std::to_string(b);
            csvPop << gen << ";" << i << ";"
                << decodeX(pop[i]) << ";"
                << std::fixed << std::setprecision(2)
                << fits[i] << ";"
                << chStr << "\n";
        }

        // Консольный прогресс
        if (gen % 10 == 0 || gen == numGens - 1) {
            std::cout << "  Ген " << std::setw(4) << gen
                << "   best_x=" << bestX
                << "   f(best)=" << std::setw(7) << std::fixed << std::setprecision(1) << bestFit
                << "   avg_f=" << std::setw(7) << std::fixed << std::setprecision(2) << avgFit
                << "   std=" << std::setw(6) << std::fixed << std::setprecision(2) << stdFit
                << "\n";
        }

        if (gen == numGens - 1) break;

        //    Новое поколение                                                    

        // VI(B): сохранить элиту
        Population newPop = eliteSelection(pop, fits, eliteCount);

        while (static_cast<int>(newPop.size()) < popSize) {
            // Выбор родителей
            int p1idx, p2idx;
            if (selChoice == 1) {
                p1idx = rouletteSelect(fits);
                int attempts = 0;
                do { p2idx = rouletteSelect(fits); ++attempts; } while (p2idx == p1idx && attempts < 20 && popSize > 1);
            }
            else {
                auto [a, b] = inbreedingSelect(pop, fits);
                p1idx = a; p2idx = b;
            }

            Chromosome c1 = pop[p1idx];
            Chromosome c2 = pop[p2idx];

            // IV. Кроссинговер
            if (prob(rng) < pCross) {
                CrossoverPair children;
                switch (crossChoice) {
                case 1: children = singlePointCrossover(c1, c2);         break;
                case 2: children = twoPointCrossover(c1, c2);            break;
                case 3: children = orderingSinglePointCrossover(c1, c2); break;
                default: children = singlePointCrossover(c1, c2);
                }
                c1 = children.first;
                c2 = children.second;
            }

            // V. Мутация
            auto mutate = [&](Chromosome& ch) {
                if (prob(rng) < pMut)
                    ch = (mutChoice == 1) ? simpleMutation(ch) : inversionMutation(ch);
                };
            mutate(c1);
            mutate(c2);

            newPop.push_back(c1);
            if (static_cast<int>(newPop.size()) < popSize)
                newPop.push_back(c2);
        }

        pop = std::move(newPop);
    }

    csvStats.close();
    csvPop.close();

    //    Итоговая таблица популяции                                             
    printPopulationTable("Итоговая популяция после " + std::to_string(numGens) + " поколений:", pop);

    //    Итог                                                                  
    const std::string sep(68, '-');
    std::cout << "\nНайденный оптимум:  x* = " << globalBestX << "\n";
    std::cout << "f(x*) = " << globalBestFit << "\n";
    std::cout << "Хромосома: ";
    for (int b : globalBestCh) std::cout << b;
    std::cout << "\n";
    std::cout << "Теор. оптимум:  x=12, f(12)=350\n";
    bool exact = (globalBestX == 12);
    std::cout << "Результат: " << (exact ? "ТОЧНЫЙ ОПТИМУМ НАЙДЕН" : "ОПТИМУМ НЕ НАЙДЕН") << "\n";
    std::cout << sep << "\n";
    std::cout << "Файлы:\n";
    std::cout << "  ga_results.csv    - статистика поколений\n";
    std::cout << "  ga_population.csv - все особи каждого поколения\n";
    system("pause");
    return 0;
}