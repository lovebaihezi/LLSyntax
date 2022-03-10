use rayon::prelude::*;

use super::{Product, Products, Symbols, G};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Display;
use std::hash::Hash;
use std::ops::{BitOr, BitOrAssign, Index, IndexMut};

#[derive(Debug)]
pub enum EliminationLeftRecursion {
    CycleProduct,
    EpsilonProduct,
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum State {
    #[default]
    Epsilon,
    Dollar,
    NonTerminal(usize),
    Terminal(usize),
}

impl Symbols for State {
    fn is_epsilon(&self) -> bool {
        matches!(self, Self::Epsilon)
    }

    fn epsilon() -> Self {
        Self::Epsilon
    }

    fn is_nonterminal(&self) -> bool {
        matches!(self, Self::NonTerminal(_))
    }

    fn is_dollar(&self) -> bool {
        matches!(self, Self::Dollar)
    }

    fn dollar() -> Self {
        Self::Dollar
    }
}

impl<T> G<T>
where
    T: std::fmt::Debug + PartialEq + Hash + Eq + Symbols + Default + Clone,
{
    pub fn new(start: T) -> Self {
        Self {
            start,
            action: Default::default(),
            map: Default::default(),
        }
    }

    pub fn nonterminals(&self) -> Vec<&T> {
        self.map.keys().collect()
    }

    pub fn is_left_recursion(&self) -> bool {
        // (out, in)
        let mut count: Vec<(usize, usize)> = Vec::with_capacity(self.map.keys().len() + 1);
        count.resize(count.capacity(), (0, 0));
        let v: Vec<_> = self.map.values().copied().collect();
        v.iter().for_each(|&i| {
            count[i].0 += self.action[i]
                .0
                .iter()
                .map(|Product(v)| v.first())
                .flatten()
                .filter(|v| v.is_nonterminal())
                .map(|v| self.map[v])
                .fold(0, |len, i| {
                    count[i].1 += 1;
                    len + 1
                });
        });
        let mut queue: VecDeque<_> = v.into_iter().filter(|&i| count[i].1 == 0).collect();
        queue.reserve(count.len());
        let mut visited = Vec::with_capacity(count.len());
        visited.resize(visited.capacity(), false);
        while !queue.is_empty() {
            let head = queue.pop_front().unwrap();
            if !visited[head] {
                visited[head] = true;
                let Products(actions) = &self.action[head];
                actions
                    .iter()
                    .map(|Product(s)| s.first())
                    .flatten()
                    .filter(|s| s.is_nonterminal())
                    .for_each(|s| {
                        queue.push_back(self.map[s]);
                        count[self.map[s]].1 -= 1
                    });
                count.get_mut(head).unwrap().0 = 0;
            }
        }
        count
            .into_iter()
            .any(|(out, r#in)| out != 0usize || r#in != 0usize)
    }

    pub fn elimination_left_recursion(&self) -> G<State> {
        debug_assert!(self.is_left_recursion());
        debug_assert!(!self.is_cycle());
        debug_assert!(!self.is_epsilon_product());
        let mut s = G::new(State::NonTerminal(0));
        let map: HashMap<&T, usize> = self
            .nonterminals()
            .iter()
            .enumerate()
            .map(|(index, &v)| (v, index))
            .collect();
        todo!();
        s
    }

    // !FIX: this is only one line different with is_left_recursion
    pub fn is_cycle(&self) -> bool {
        // (out, in)
        let mut count: Vec<(usize, usize)> = Vec::with_capacity(self.map.keys().len() + 1);
        count.resize(count.capacity(), (0, 0));
        let v: Vec<_> = self.map.values().copied().collect();
        v.iter().for_each(|&i| {
            count[i].0 += self.action[i]
                .0
                .iter()
                .filter(|Product(v)| v.len() == 1)
                .map(|Product(v)| unsafe { v.first().unwrap_unchecked() })
                .filter(|v| v.is_nonterminal())
                .map(|v| self.map[v])
                .fold(0, |len, i| {
                    count[i].1 += 1;
                    len + 1
                });
        });
        let mut queue: VecDeque<_> = v.into_iter().filter(|&i| count[i].1 == 0).collect();
        queue.reserve(count.len());
        let mut visited = Vec::with_capacity(count.len());
        visited.resize(visited.capacity(), false);
        while !queue.is_empty() {
            let head = queue.pop_front().unwrap();
            if !visited[head] {
                visited[head] = true;
                let Products(actions) = &self.action[head];
                actions
                    .iter()
                    .map(|Product(s)| s.first())
                    .flatten()
                    .filter(|s| s.is_nonterminal())
                    .for_each(|s| {
                        queue.push_back(self.map[s]);
                        count[self.map[s]].1 -= 1
                    });
                count.get_mut(head).unwrap().0 = 0;
            }
        }
        count
            .into_iter()
            .any(|(out, r#in)| out != 0usize || r#in != 0usize)
    }

    pub fn is_epsilon_product(&self) -> bool {
        self.action.iter().any(|Products(actions)| {
            actions.iter().any(|Product(action)| {
                action.len() == 1 && action.iter().any(|v| v == &T::epsilon())
            })
        })
    }

    // TODO: Make a cache for this!
    pub fn first(&self, sign: T) -> HashSet<T> {
        if sign.is_epsilon() {
            [T::epsilon()].into_iter().collect()
        } else if sign.is_terminal() {
            [sign].into_iter().collect()
        } else {
            let mut set = HashSet::new();
            let Products(products) = &self.action[self.map[&sign]];
            for Product(product) in products {
                for v in product {
                    let s = self.first(v.clone());
                    let flag = !s.contains(&T::epsilon());
                    s.into_iter().for_each(|v| {
                        set.insert(v);
                    });
                    if flag {
                        break;
                    }
                }
            }
            set
        }
    }

    // TODO: Make a cache to store every product will or not
    pub fn will_product_epslion(&self, sign: &T) -> bool {
        if sign.is_nonterminal() {
            let Products(products) = &self.action[self.map[sign]];
            let mut queue: VecDeque<&T> = Default::default();
            for v in products.iter().map(|Product(v)| v.first()).flatten() {
                if v.is_epsilon() {
                    return true;
                } else if v.is_nonterminal() {
                    queue.push_back(v)
                }
            }
            while !queue.is_empty() {
                let top = queue.pop_front().unwrap();
                let Products(v) = &self.action[self.map[top]];
                if v.iter()
                    .map(|Product(v)| v.first())
                    .flatten()
                    .any(|v| v.is_epsilon())
                {
                    return true;
                } else {
                    v.iter()
                        .map(|Product(v)| v.first())
                        .flatten()
                        .filter(|v| v.is_nonterminal())
                        .for_each(|v| queue.push_back(v));
                }
            }
            false
        } else {
            false
        }
    }

    pub fn follow(&self, sign: T) -> HashSet<T> {
        if sign.is_nonterminal() {
            let mut set = HashSet::new();
            if sign == self.start {
                set.insert(T::dollar());
            }
            for (from, vec) in self
                .action
                .iter()
                .enumerate()
                .map(|(index, Products(v))| -> (_, Vec<_>) {
                    (
                        self.map.iter().find(|(_, i)| **i == index).unwrap().0,
                        v.iter().filter(|Product(v)| v.contains(&sign)).collect(),
                    )
                })
                .filter(|(from, v)| !v.is_empty() && *from != &sign)
            {
                vec.iter().for_each(|Product(v)| {
                    match v.iter().skip_while(|&v| v == &sign).nth(2) {
                        Some(v) => {
                            let s = self.first(v.clone());
                            if s.contains(&T::epsilon()) {
                                self.follow(from.clone())
                                    .into_iter()
                                    .chain(s.into_iter())
                                    .filter(|v| !v.is_epsilon())
                                    .for_each(|v| {
                                        set.insert(v);
                                    });
                            } else {
                                s.into_iter().for_each(|v| {
                                    set.insert(v);
                                });
                            }
                        }
                        None => self.follow(from.clone()).into_iter().for_each(|v| {
                            set.insert(v);
                        }),
                    }
                })
            }
            set
        } else {
            Default::default()
        }
    }

    pub fn generate_predictive_parsing(&self) -> HashMap<T, HashMap<T, (usize, usize)>> {
        let mut map: HashMap<_, HashMap<_, _>> = HashMap::with_capacity(self.action.len());
        let mut index_t: Vec<&T> = Vec::with_capacity(self.map.len());
        let x = T::epsilon();
        index_t.resize(self.map.len(), &x);
        self.map.iter().for_each(|(value, key)| {
            index_t[*key] = value;
        });
        for (index, Products(products)) in self.action.iter().enumerate() {
            let v = index_t[index];
            for (i, Product(product)) in products.iter().enumerate() {
                let first = self.first(product.first().unwrap().clone());
                let flag = first.contains(&T::epsilon());
                map.entry(v.clone())
                    .or_insert_with(|| Default::default())
                    .extend(
                        first
                            .into_iter()
                            .filter(|v| v.is_terminal())
                            .map(|v| (v, (index, i))),
                    );
                if flag {
                    let follow = self.follow(v.clone());
                    if follow.contains(&T::dollar()) {
                        map.entry(v.clone())
                            .or_insert_with(|| Default::default())
                            .insert(T::dollar(), (index, i));
                    }
                    map.entry(v.clone())
                        .or_insert_with(|| Default::default())
                        .extend(
                            follow
                                .into_iter()
                                .filter(|v| v.is_terminal())
                                .map(|v| (v, (index, i))),
                        );
                }
            }
        }
        map
    }
    fn check(&self, iter: &mut impl Iterator<Item = T>) -> Result<Vec<(usize, usize)>, String>
    where
        T: Display,
    {
        let mut stack = VecDeque::new();
        let table = self.generate_predictive_parsing();
        stack.push_back(T::dollar());
        stack.push_back(self.start.clone());
        let mut a = iter.next().ok_or_else(|| "iter is empty!".to_string())?;
        let mut result = Vec::with_capacity(iter.size_hint().0);
        while !stack.is_empty() {
            let top = unsafe { stack.pop_back().unwrap_unchecked() };
            if top == a {
                // a = iter.next().ok_or_else(|| "iter should not end here")?;
                a = match iter.next() {
                    Some(v) => v,
                    None => T::dollar(),
                }
            } else if top.is_terminal() {
                return Err(format!("terminal {:?} not found!", a));
            } else {
                if let Some((x, y)) = table.get(&top).and_then(|map| map.get(&a)) {
                    print!("{}  ", a);
                    print!("[");
                    for i in &stack {
                        print!("{} ", i);
                    }
                    print!("] ");
                    println!("{} -> {}", &top, self.action[*x].0[*y]);
                    result.push((*x, *y));
                    stack.extend(
                        self.action[*x].0[*y]
                            .0
                            .iter()
                            .filter(|v| !v.is_epsilon())
                            .map(|v| v.clone())
                            .rev(),
                    )
                } else {
                    return Err(format!("no next step for current [{:?}][{:?}]", top, a));
                }
            }
        }
        Ok(result)
    }
}

impl<T> Index<T> for G<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Copy + Eq + Symbols,
{
    fn index(&self, index: T) -> &Self::Output {
        &self.action[self.map[&index]]
    }

    type Output = Products<T>;
}

impl<T> IndexMut<T> for G<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Copy + Eq + Symbols + Default,
{
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        let len = self.action.len();
        let index = *self.map.entry(index).or_insert_with(|| len);
        self.action.push(Default::default());
        &mut self.action[index]
    }
}

impl<T> BitOrAssign<Products<T>> for Products<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Copy + Eq + Symbols,
{
    fn bitor_assign(&mut self, mut rhs: Products<T>) {
        self.0.append(&mut rhs.0)
    }
}

impl<T> BitOr for Products<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Eq + Symbols + Default,
{
    type Output = Products<T>;

    fn bitor(mut self, mut rhs: Self) -> Self::Output {
        self.0.append(&mut rhs.0);
        self
    }
}

impl<T> Display for Products<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Eq + Symbols + Default + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for v in self.0.iter().take(self.0.len() - 1) {
            f.write_fmt(format_args!("{} | ", v))?;
        }
        if let Some(v) = self.0.last() {
            f.write_fmt(format_args!("{}", v))?;
        }
        Ok(())
    }
}

impl<T> Display for Product<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Eq + Symbols + Default + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for v in self.0.iter().take(self.0.len() - 1) {
            f.write_fmt(format_args!("{} ", v))?;
        }
        if let Some(v) = self.0.last() {
            f.write_fmt(format_args!("{}", v))?;
        }
        Ok(())
    }
}

impl<T> Display for G<T>
where
    T: std::fmt::Debug + PartialEq + Hash + Eq + Symbols + Default + Clone + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in self.nonterminals() {
            f.write_fmt(format_args!("{:?} -> {}\n", i, self.action[self.map[i]]))?;
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! A {
    [$($x:expr),*$(,)*] => {
        $crate::grammar::Products(vec![Product(vec![$($x,)*])])
    };
}

#[macro_export]
macro_rules! T {
    ($e:expr) => {
        Terminal($e)
    };
}

#[cfg(test)]
mod grammar_implement_test {
    use std::collections::HashSet;

    use crate::{
        grammar::{Product, Symbols, G},
        Symbol,
    };
    Symbol! { Test1, S, B, D, E, }
    #[test]
    fn terminal_and_nonterminal() {
        assert!(!Test1::<char>::S.is_terminal());
        assert!(!Test1::<char>::Epsilon.is_terminal());
        assert!(!Test1::<char>::Epsilon.is_nonterminal());
    }
    #[test]
    fn operator_overwrite() {
        use Test1::*;
        let mut s = G::new(S);
        s[S] |= A![Terminal('a')] | A![Terminal('a'), B];
        s[B] |= A![B];
        assert!(s.is_cycle());
        assert!(s.is_left_recursion());
    }
    #[test]
    fn no_left_recursion() {
        use Test1::*;
        let mut s = G::new(S);
        s[S] |= A![Terminal('a')];
        assert!(!s.is_left_recursion());
    }
    #[test]
    fn is_left_recursion() {
        use Test1::*;
        let mut s = G::new(S);
        s[S] |= A![B] | A![Terminal('a')];
        s[B] |= A![D];
        s[D] |= A![E];
        s[E] |= A![S];
        assert!(s.is_cycle());
        assert!(s.is_left_recursion());
    }
    #[test]
    fn is_cycle() {
        use Test1::*;
        let mut s = G::new(S);
        s[S] |= A![B] | A![Terminal('a')];
        s[B] |= A![D];
        s[D] |= A![S];
        assert!(s.is_left_recursion());
        assert!(!s.is_epsilon_product());
        assert!(s.is_cycle());
    }
    #[test]
    fn is_epsilon_product() {
        use Test1::*;
        let mut s = G::new(S);
        s[S] |= A![Terminal('a')] | A![B];
        s[B] |= A![Terminal('c')];
        assert!(!s.is_cycle());
        assert!(!s.is_left_recursion());
        assert!(!s.is_epsilon_product());
        let mut s = G::new(S);
        s[S] |= A![Terminal(1)];
        s[B] |= A![Epsilon];
        assert!(s.is_epsilon_product());
    }
    #[test]
    #[ignore = "elimination_left_recursion not finished!"]
    fn elimination_left_recursion() {
        use Test1::*;
        let mut s = G::new(S);
        s[S] |= A![S, T!('a'), T!('b'), B] | A![B, T!('x')];
        s[B] |= A![T!('o'), B];
        assert!(s.is_left_recursion());
        assert!(!s.is_cycle());
        assert!(!s.is_epsilon_product());
        let x = s.elimination_left_recursion();
        assert!(!x.is_epsilon_product());
        assert!(!x.is_cycle());
        assert!(!x.is_left_recursion());
    }
    #[test]
    fn will_product_epslion() {
        use Test1::*;
        let mut s = G::new(S);
        s[S] |= A![T!(1)] | A![Epsilon];
        assert!(s.will_product_epslion(&S));
        let mut s = G::new(S);
        s[S] |= A![T!(1)];
        assert!(!s.will_product_epslion(&S));
    }
    #[test]
    fn first_set() {
        Symbol! {Test, E, E1, T, T1, F, };
        use Test::*;
        let mut s = G::new(E);
        s[E] |= A![T, E1];
        s[E1] |= A![T!('+'), T, E1] | A![Epsilon];
        s[T] |= A![F, T1];
        s[T1] |= A![T!('*'), F, T1] | A![Epsilon];
        s[F] |= A![T!('('), E, T!(')')] | A![T!('x')];
        assert_eq!(s.first(E), [T!('('), T!('x')].into_iter().collect());
        assert_eq!(s.first(E1), [T!('+'), Epsilon].into_iter().collect());
        assert_eq!(s.first(T1), [T!('*'), Epsilon].into_iter().collect());
        let mut s = G::new(E);
        s[E] |= A![T] | A![Epsilon];
        s[T] |= A![T!('a')] | A![Epsilon];
        assert_eq!(s.first(T), [T!('a'), Epsilon].into_iter().collect());
        let mut s = G::<Test1<char>>::new(Test1::<char>::E);
        s[Test1::<char>::E] |= A![Test1::<char>::Epsilon];
        assert_eq!(
            s.first(Test1::<char>::E),
            [Test1::<char>::Epsilon].into_iter().collect()
        );
    }
    #[test]
    fn follow_set() {
        Symbol! {Test, E, E1, T, T1, F, };
        use Test::*;
        let mut s = G::new(E);
        s[E] |= A![T, E1];
        s[E1] |= A![T!('+'), T, E1] | A![Epsilon];
        s[T] |= A![F, T1];
        s[T1] |= A![T!('*'), F, T1] | A![Epsilon];
        s[F] |= A![T!('('), E, T!(')')] | A![T!('x')];
        assert_eq!(s.follow(E), [T!(')'), Test::dollar()].into_iter().collect());
        assert_eq!(s.follow(E), s.follow(E1));
    }
    #[test]
    fn predictive_table() {
        Symbol! {Test, E, E1, T, T1, F, };
        use Test::*;
        let mut s = G::new(E);
        s[E] |= A![T, E1];
        s[E1] |= A![T!('+'), T, E1] | A![Epsilon];
        s[T] |= A![F, T1];
        s[T1] |= A![T!('*'), F, T1] | A![Epsilon];
        s[F] |= A![T!('('), E, T!(')')] | A![T!('x')] | A![T!('y')];
        println!("{}", s);
        let mut all_keys: HashSet<_> = HashSet::new();
        s.generate_predictive_parsing()
            .values()
            .for_each(|v| all_keys.extend(v.keys()));
        print!("|  |");
        for i in &all_keys {
            print!("{:^12}|", format!("{}", i));
        }
        println!();
        for (t, map) in s.generate_predictive_parsing() {
            print!("|{:2}|", format!("{}", t));
            for i in &all_keys {
                print!(
                    "{:^12}|",
                    match map.get(i) {
                        Some((x, y)) => format!("{} -> {}", t, s.action[*x].0[*y]),
                        None => "".to_string(),
                    }
                );
            }
            println!();
        }
    }
    #[test]
    fn check() {
        Symbol! {Test, E, E1, T, T1, F, };
        use Test::*;
        let mut s = G::new(E);
        s[E] |= A![T, E1];
        s[E1] |= A![T!('+'), T, E1] | A![Epsilon];
        s[T] |= A![F, T1];
        s[T1] |= A![T!('*'), F, T1] | A![Epsilon];
        s[F] |= A![T!('('), E, T!(')')] | A![T!('x')] | A![T!('y')];
        println!("{}", s);
        let x = "x+y*x";
        let mut iter = x.chars().map(|v| T!(v));
        let result = s.check(&mut iter);
        assert!(result.is_ok());
    }
    #[test]
    fn home_work() {
        Symbol! {Test, S, D,};
        use Test::*;
        let mut s = G::new(S);
        s[S] |= A![T!('r'), D];
        s[D] |= A![T!('i'), T!(','), D] | A![T!('i')];
        println!("{}", s);
        let mut all_keys: HashSet<Test<_>> = HashSet::new();
        s.generate_predictive_parsing()
            .values()
            .for_each(|v| all_keys.extend(v.keys()));
        print!("|  |");
        for i in &all_keys {
            print!("{:^12}|", format!("{}", i));
        }
        println!();
        for i in &all_keys {
            println!(
                "{:?} | {:?} | {:?} | {:?}",
                s.first(S),
                s.follow(S),
                s.first(D),
                s.follow(D)
            );
        }
        println!();
        for (t, map) in s.generate_predictive_parsing() {
            print!("|{:2}|", format!("{}", t));
            for i in &all_keys {
                print!(
                    "{:^12}|",
                    match map.get(i) {
                        Some((x, y)) => format!("{} -> {}", t, s.action[*x].0[*y]),
                        None => "".to_string(),
                    }
                );
            }
            println!();
        }
    }
}
