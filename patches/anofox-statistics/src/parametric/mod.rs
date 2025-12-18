pub mod anova;
pub mod levene;
pub mod ttest;
pub mod yuen;

pub use anova::{
    one_way_anova, repeated_measures_anova, two_way_anova, AnovaKind, AnovaTableRow,
    CorrectedResult, OneWayAnovaResult, RmAnovaResult, SphericityResult, TwoWayAnovaResult,
};
pub use levene::{brown_forsythe, LeveneResult};
pub use ttest::{t_test, Alternative, TTestKind, TTestResult};
pub use yuen::{yuen_test, YuenConfInt, YuenResult};
