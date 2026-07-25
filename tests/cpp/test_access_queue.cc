// Unit tests for remodel::AccessQueue.
//
// The class is linked from the object file that the simulator build produces,
// so these cases pin the behaviour of the production code itself rather than of
// a copy.

#include <gtest/gtest.h>

#include "access_queue.h"

namespace {

using remodel::AccessQueue;

// AccessQueue stores mem_access_t pointers and never dereferences them, so
// distinct non-null sentinels are enough to observe ordering and occupancy.
// Its destructor deletes whatever is still queued (it owns what it holds), so
// every test drains the queue before it leaves scope.
mem_access_t *sentinel(unsigned int index) {
  return reinterpret_cast<mem_access_t *>(0x1000 + 0x10 * (index + 1));
}

TEST(AccessQueue, StartsEmpty) {
  AccessQueue queue(4);
  EXPECT_TRUE(queue.empty());
  EXPECT_FALSE(queue.full());
  EXPECT_EQ(0u, queue.size());
}

TEST(AccessQueue, ServesAccessesInPushOrder) {
  AccessQueue queue(4);
  queue.push(sentinel(0));
  queue.push(sentinel(1));
  queue.push(sentinel(2));

  EXPECT_EQ(sentinel(0), queue.front());
  queue.pop();
  EXPECT_EQ(sentinel(1), queue.front());
  queue.pop();
  EXPECT_EQ(sentinel(2), queue.front());
  queue.pop();
  EXPECT_TRUE(queue.empty());
}

TEST(AccessQueue, FrontDoesNotConsume) {
  AccessQueue queue(4);
  queue.push(sentinel(0));
  EXPECT_EQ(sentinel(0), queue.front());
  EXPECT_EQ(sentinel(0), queue.front());
  EXPECT_EQ(1u, queue.size());
  queue.pop();
}

TEST(AccessQueue, SizeAndEmptyTrackPushAndPop) {
  AccessQueue queue(4);
  for (unsigned int i = 0; i < 3; i++) {
    queue.push(sentinel(i));
    EXPECT_EQ(i + 1, queue.size());
    EXPECT_FALSE(queue.empty());
  }
  for (unsigned int i = 3; i > 0; i--) {
    EXPECT_EQ(i, queue.size());
    queue.pop();
  }
  EXPECT_EQ(0u, queue.size());
  EXPECT_TRUE(queue.empty());
}

TEST(AccessQueue, BecomesFullAtItsCapacityAndFreesUpOnPop) {
  AccessQueue queue(2);
  EXPECT_FALSE(queue.full());
  queue.push(sentinel(0));
  EXPECT_FALSE(queue.full());
  queue.push(sentinel(1));
  EXPECT_TRUE(queue.full());
  EXPECT_EQ(2u, queue.size());

  queue.pop();
  EXPECT_FALSE(queue.full());
  queue.push(sentinel(2));
  EXPECT_TRUE(queue.full());

  queue.pop();
  queue.pop();
  EXPECT_TRUE(queue.empty());
}

// A capacity of zero makes `empty()` and `full()` true at the same time, since
// `full()` only compares the size against the bound. Pinned as observed.
TEST(AccessQueue, ZeroCapacityIsBothEmptyAndFull) {
  AccessQueue queue(0);
  EXPECT_TRUE(queue.empty());
  EXPECT_TRUE(queue.full());
  EXPECT_EQ(0u, queue.size());
}

// The bounds and emptiness preconditions are plain asserts, and the simulator
// is built without NDEBUG, so violating them aborts rather than corrupting the
// queue. These cases pin that the guard is live in the shipped objects.
TEST(AccessQueueDeathTest, PushingPastCapacityAborts) {
  AccessQueue queue(1);
  queue.push(sentinel(0));
  ASSERT_TRUE(queue.full());
  EXPECT_DEATH(queue.push(sentinel(1)), "m_max_size");
  queue.pop();
}

TEST(AccessQueueDeathTest, PoppingAnEmptyQueueAborts) {
  AccessQueue queue(4);
  EXPECT_DEATH(queue.pop(), "m_accesses.empty");
}

TEST(AccessQueueDeathTest, ReadingTheFrontOfAnEmptyQueueAborts) {
  AccessQueue queue(4);
  EXPECT_DEATH(queue.front(), "m_accesses.empty");
}

}  // namespace
