use super::{Product, Products, Symbols, G};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::ops::{BitOr, BitOrAssign, Index, IndexMut};
#[derive(Debug)]
pub enum EliminationLeftRecursion {
    CycleProduct,
    epsilonProduct,
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum State {
    #[default]
    Epsilon,
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
}

impl<T> G<T>
where
    T: std::fmt::Debug + PartialEq + Hash + Eq + Symbols + Default + Clone,
{
    pub fn new(start: T) -> Self {
        let cache = Vec::new();
        let action = Vec::new();
        Self {
            start,
            cache,
            action,
            map: HashMap::new(),
        }
    }

    pub fn nonterminals(&self) -> Vec<&T> {
        self.map.keys().collect()
    }

    fn is_left_recursion(&self) -> bool {
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

    fn elimination_left_recursion(&self) -> G<State> {
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
    fn is_cycle(&self) -> bool {
        // (out, in)
        let mut count: Vec<(usize, usize)> = Vec::with_capacity(self.map.keys().len() + 1);
        count.resize(count.capacity(), (0, 0));
        let v: Vec<_> = self.map.values().copied().collect();
        v.iter().for_each(|&i| {
            count[i].0 += self.action[i]
                .0
                .iter()
                .filter(|Product(v)| v.len() == 1)
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

    fn is_epsilon_product(&self) -> bool {
        self.action.iter().any(|Products(actions)| {
            actions.iter().any(|Product(action)| {
                action.len() == 1 && action.iter().any(|v| v == &T::epsilon())
            })
        })
    }

    fn first(&self, sign: T) -> HashSet<T> {
        assert!(!sign.is_epsilon());
        if sign.is_terminal() {
            [sign].into_iter().collect()
        } else if let Some(v) = self.cache.get(self.map[&sign]) {
            v[0].clone()
        } else {
            let mut set = HashSet::new();
            let Products(actions) = &self.action[self.map[&sign]];
            for Product(action) in actions {
                for y in action {
                    if y.is_terminal() {
                        set.insert(y.clone());
                    }
                }
            }
            set
        }
    }

    fn follow(&self, sign: T) -> HashSet<&T> {
        todo!()
    }
}

impl<T> Index<T> for G<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Copy + Eq + Symbols + Default,
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
    T: std::fmt::Debug + Clone + PartialEq + Hash + Copy + Eq + Symbols + Default,
{
    fn bitor_assign(&mut self, mut rhs: Products<T>) {
        self.0.append(&mut rhs.0)
    }
}

impl<T> BitOr for Products<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Copy + Eq + Symbols + Default,
{
    type Output = Products<T>;

    fn bitor(mut self, mut rhs: Self) -> Self::Output {
        self.0.append(&mut rhs.0);
        self
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
    use crate::{
        grammar::{Product, Symbols, G},
        Symbol,
    };
    Symbol! { Test1, S, B, D, E, }
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
    fn elimination_left_recursion() {
        use Test1::*;
        let mut s = G::new(S);
        s[S] |= A![S, T!('a'), T!('b'), B] | A![B, T!('x')];
        s[B] |= A![T!('o'), B];
        assert!(s.is_left_recursion());
        assert!(!s.is_cycle());
        assert!(!s.is_epsilon_product());
        let x = s.elimination_left_recursion();
        assert!(!s.is_epsilon_product());
        assert!(!s.is_cycle());
        assert!(!s.is_left_recursion());
    }
}
