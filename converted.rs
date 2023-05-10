

// Copyright 2022-23, Juspay India Pvt Ltd
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program
// is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details. You should have received a copy of
// the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

#![allow(deprecated)]

use crate::cache::{CacheFlow, HedisFlow};
use crate::db_flow::{EsqDBFlow, EsqTransactionable};
use crate::domain::types::{Exophone, Merchant};
use crate::storage::queries::exophone as queries;
use crate::storage::queries::CacheConfig;
use crate::utils::{common, id};
use std::collections::HashMap;
use std::option::Option::Some;
use std::sync::Arc;
use std::{thread, time};
use tokio::sync::RwLock;
use tracing::instrument;

pub async fn find_all_by_merchant_id<Q>(
    merchant_id: id::Id<Merchant, Q>,
    cache_configs: Arc<CacheConfig>,
    db_flow: Arc<dyn EsqDBFlow<Q>>,
    cache_flow: Arc<dyn CacheFlow<Q>>,
) -> Vec<Exophone>
where
    Q: HedisFlow + EsqTransactionable,
{
    match cache_flow.get(EXOPHONE_KEY_PREFIX.to_owned() + &merchant_id.get_id()).await {
        Some(result) => result,
        None => {
            let result = queries::find_all_by_merchant_id(merchant_id.clone(), db_flow.as_ref()).await.expect("Failed to fetch exophones by merchant_id");
            if let Some(exophone) = result.first().cloned() {
                cache_flow.set(
                    EXOPHONE_KEY_PREFIX.to_owned() + &merchant_id.get_id(),
                    result.clone(),
                    cache_configs.exp_time,
                ).await
                .expect("Failed to cache exophones by merchant_id");
            }
            result
        }
    }
}

pub async fn find_by_phone<Q>(
    phone: &str,
    cache_configs: Arc<CacheConfig>,
    db_flow: Arc<dyn EsqDBFlow<Q>>,
    cache_flow: Arc<dyn CacheFlow<Q>>,
) -> Option<Exophone>
where
    Q: HedisFlow + EsqTransactionable,
{
    let exophones = find_all_by_phone(phone, cache_configs.clone(), db_flow.clone(), cache_flow.clone()).await;
    exophones.into_iter().find(|exophone| exophone.primary_phone == phone || exophone.backup_phone == phone)
}

pub async fn find_by_primary_phone<Q>(
    phone: &str,
    cache_configs: Arc<CacheConfig>,
    db_flow: Arc<dyn EsqDBFlow<Q>>,
    cache_flow: Arc<dyn CacheFlow<Q>>,
) -> Option<Exophone>
where
    Q: HedisFlow + EsqTransactionable,
{
    let exophones = find_all_by_phone(phone, cache_configs.clone(), db_flow.clone(), cache_flow.clone()).await;
    exophones.into_iter().find(|exophone| exophone.primary_phone == phone)
}

pub async fn find_all_by_phone<Q>(
    phone: &str,
    cache_configs: Arc<CacheConfig>,
    db_flow: Arc<dyn EsqDBFlow<Q>>,
    cache_flow: Arc<dyn CacheFlow<Q>>,
) -> Vec<Exophone>
where
    Q: HedisFlow + EsqTransactionable,
{
    match cache_flow.get(phone.to_owned()).await {
        Some(merchant_id) => match cache_flow.get(EXOPHONE_KEY_PREFIX.to_owned() + &merchant_id).await {
            Some(result) => result,
            None => {
                let result = queries::find_all_by_merchant_id(
                    id::Id::new(merchant_id.parse::<u64>().unwrap()),
                    db_flow.as_ref(),
                )
                .await
                .expect("Failed to fetch exophones by phone");
                if let Some(exophone) = result.first().cloned() {
                    cache_flow
                        .set(
                            EXOPHONE_KEY_PREFIX.to_owned() + &merchant_id,
                            result.clone(),
                            cache_configs.exp_time,
                        )
                        .await
                        .expect("Failed to cache exophones by phone");
                }
                result
            }
        },
        None => {
            let result = queries::find_all_by_phone(phone, db_flow.as_ref()).await.expect("Failed to fetch exophones by phone");
            if let Some(exophone) = result.first().cloned() {
                let mut cache_payload = HashMap::new();
                cache_payload.insert(phone.to_owned(), exophone.merchant_id.to_string());
                cache_payload.insert(EXOPHONE_KEY_PREFIX.to_owned() + &exophone.merchant_id.to_string(), serde_json::to_string(&result.clone()).unwrap());
                cache_flow
                    .multi_set(cache_payload, cache_configs.exp_time)
                    .await
                    .expect("Failed to multi_cache exophones");
            }
            result
        }
    }
}

pub async fn find_all_exophones<Q>(db_flow: Arc<dyn EsqDBFlow<Q>>) -> Vec<Exophone>
where
    Q: EsqTransactionable,
{
    queries::find_all_exophones(db_flow.as_ref()).await.expect("Failed to fetch all exophones")
}

#[instrument(skip(cache_flow), err)]
pub async fn clear_cache<Q>(
    merchant_id: id::Id<Merchant, Q>,
    exophones: &[Exophone],
    cache_flow: Arc<dyn CacheFlow<Q>>,
) where
    Q: HedisFlow,
{
    cache_flow.del(&[EXOPHONE_KEY_PREFIX.to_owned() + &merchant_id.get_id()]).await;
    for exophone in exophones {
        cache_flow
            .del(&[exophone.primary_phone.clone(), exophone.backup_phone.clone()])
            .await;
    }
}

#[instrument(skip(cache_flow), err)]
pub async fn clear_all_cache<Q>(cache_flow: Arc<dyn CacheFlow<Q>>) where Q: HedisFlow {
    cache_flow.del_by_pattern(&EXOPHONE_PATTERN).await;
}

pub async fn create<Q>(exophone: Exophone, db_flow: Arc<dyn EsqDBFlow<Q>>) -> Result<(), sqlx::Error> {
    queries::create(exophone, db_flow.as_ref()).await
}

pub async fn update_affected_phones<Q>(phones: Vec<String>, db_flow: Arc<dyn EsqDBFlow<Q>>) -> Result<(), sqlx::Error> {
    queries::update_affected_phones(phones, db_flow.as_ref()).await
}

pub async fn delete_by_merchant_id<Q>(merchant_id: id::Id<Merchant, Q>, db_flow: Arc<dyn EsqDBFlow<Q>>) -> Result<(), sqlx::Error> {
    queries::delete_by_merchant_id(merchant_id, db_flow.as_ref()).await
}

const EXOPHONE_PATTERN: &str = "CachedQueries:Exophones:*";
const EXOPHONE_KEY_PREFIX: &str = "CachedQueries:Exophones:MerchantId-";
