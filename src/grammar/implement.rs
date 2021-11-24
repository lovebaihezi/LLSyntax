use super::{Action, Actions, Construct, Symbols, G};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::ops::{BitOr, BitOrAssign, Index, IndexMut};

#[derive(Debug)]
pub enum EliminationLeftRecursion {
    CycleProduct,
    EpilisonProduct,
}

impl<T> G<T>
where
    T: std::fmt::Debug + PartialEq + Hash + Eq + Symbols + Default + Clone,
{
    pub fn new(start: T, nonterminal_len: usize) -> Self {
        let cache = Vec::with_capacity(nonterminal_len);
        let action = Vec::with_capacity(nonterminal_len);
        Self {
            start,
            cache,
            action,
            map: HashMap::with_capacity(nonterminal_len),
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
                 .0
                .iter()
                .map(|Action(v)| v.first())
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
                let Construct(Actions(actions)) = &self.action[head];
                actions
                    .iter()
                    .map(|Action(s)| s.first())
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

    pub fn elimination_left_recursion(&mut self) {
        debug_assert!(self.is_left_recursion());
        //debug_assert!(!self.is_cycle());
        debug_assert!(!self.is_epilison_product());
        let actions = self.nonterminals();
        for i in 0..actions.len() {
            for j in 0..i {}
        }
    }

    pub fn is_epilison_product(&self) -> bool {
        self.action.iter().any(|Construct(Actions(actions))| {
            actions.iter().any(|Action(action)| {
                action.len() == 1 && action.iter().any(|v| v == &T::epilison())
            })
        })
    }

    pub fn elimination_cycle(&mut self) {
        todo!()
    }

    pub fn elimination_epilison(&mut self) {
        todo!()
    }

    pub fn left_factoring(&mut self) {
        todo!()
    }

    pub fn first(&mut self, sign: T) -> HashSet<&T>
    where
        T: Clone,
    {
        todo!()
    }

    pub fn follow(&self, sign: T) -> HashSet<&T>
    where
        T: Clone,
    {
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

    type Output = Construct<T>;
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

impl<T> BitOrAssign<Actions<T>> for Construct<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Copy + Eq + Symbols + Default,
{
    fn bitor_assign(&mut self, mut rhs: Actions<T>) {
        self.0 .0.append(&mut rhs.0)
    }
}

impl<T> BitOr for Actions<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Hash + Copy + Eq + Symbols + Default,
{
    type Output = Actions<T>;

    fn bitor(mut self, mut rhs: Self) -> Self::Output {
        self.0.append(&mut rhs.0);
        self
    }
}

#[macro_export]
macro_rules! A {
    [$($x:expr),*$(,)*] => {
        $crate::grammar::Actions(vec![Action(vec![$($x,)*])])
    };
}

#[cfg(test)]
mod grammar_implement_test {
    use crate::{
        grammar::{Action, Symbols, G},
        Symbol,
    };
    Symbol! { Test1, S, B, D, E, }
    #[test]
    fn operator_overwrite() {
        use Test1::*;
        let mut s = G::new(S, 2);
        s[S] |= A![Terminal('a')] | A![Terminal('a'), B];
        s[B] |= A![B];
        // assert!(s.is_cycle());
        assert!(s.is_left_recursion());
    }
    #[test]
    fn no_left_rescursion() {
        use Test1::*;
        let mut s = G::new(S, 1);
        s[S] |= A![Terminal('a')];
        assert!(!s.is_left_recursion());
    }
    #[test]
    fn is_left_recursion() {
        use Test1::*;
        let mut s = G::new(S, 4);
        s[S] |= A![B] | A![Terminal('a')];
        s[B] |= A![D];
        s[D] |= A![E];
        s[E] |= A![S];
        // assert!(s.is_cycle());
        assert!(s.is_left_recursion());
    }
    #[test]
    fn is_cycle() {
        use Test1::*;
        let mut s = G::new(S, 3);
        s[S] |= A![B] | A![Terminal('a')];
        s[B] |= A![D];
        s[D] |= A![S];
        assert!(s.is_left_recursion());
        assert!(!s.is_epilison_product());
        // assert!(s.is_cycle());
    }
    #[test]
    fn is_epilison_product() {
        use Test1::*;
        let mut s = G::new(S, 2);
        s[S] |= A![Terminal('a')] | A![B];
        s[B] |= A![Terminal('c')];
        // assert!(!s.is_cycle());
        assert!(!s.is_left_recursion());
        assert!(!s.is_epilison_product());
        let mut s = G::new(S, 2);
        s[S] |= A![Terminal(1)];
        s[B] |= A![Epilison];
        assert!(s.is_epilison_product());
    }
    use std::{fmt::Debug, hash::Hash};
    #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    enum Symbol {
        NonTerminal(usize),
        Terminal(char),
        Epilison,
    }
    #[test]
    fn grammar_first() {}
}
