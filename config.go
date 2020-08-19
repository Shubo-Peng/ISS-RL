/*
Copyright IBM Corp. All Rights Reserved.

SPDX-License-Identifier: Apache-2.0
*/

package mirbft

import (
	pb "github.com/IBM/mirbft/mirbftpb"
)

type Config struct {
	// ID is the NodeID for this instance.
	ID uint64

	// Logger provides the logging functions.
	Logger Logger

	// BatchParameters determines under what conditions the queued
	// pieces of data should be converted into a batch and consented on
	BatchParameters BatchParameters

	// HeartbeatTicks is the number of ticks before a heartbeat is emitted
	// by a leader.
	HeartbeatTicks int

	// SuspectTicks is the number of ticks a bucket may not progress before
	// the node suspects the epoch has gone bad.
	SuspectTicks int

	// NewEpochTimeoutTicks is the number of ticks a replica will wait until
	// it suspects the epoch leader has failed.  This value must be greater
	// than 1, as rebroadcast ticks are computed as half this value.
	NewEpochTimeoutTicks int

	// BufferSize is the number of messages buffered waiting for this node
	// to process. If buffer is full, oldest message will be dropped to
	// make room for new messages
	BufferSize int

	// Introspector, if set, has its Selected method invoked each time the
	// state machine undergoes some mutation.  This allows for additional
	// external insight into the state machine, but comes at a performance cost
	// and would generally not be enabled outside of a test or debug setting.
	Introspector Introspector
}

type BatchParameters struct {
	BatchSize int
}

// Introspector provides a way for a consumer to gain insight into
// the internal operation of the state machine.  And is usually not
// interesting outside of debugging or testing scenarios.
type Introspector interface {
	// Inspect is invoked for each mutating state machine event
	Inspect(s *pb.StateEvent)
}
